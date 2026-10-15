"""Planners: the component that decides which tool to call, and with what.

OllamaPlanner  Model-driven. Talks to a local open-weights model (Qwen by
               default) through Ollama's OpenAI-compatible API using tool
               calling. This is the real agent.
ScriptedPlanner  Deterministic regex baseline with NO model. Used for CI and as
                 an offline fallback. Results from it must never be presented as
                 model-driven results.

Both return {"content": str | None, "tool_calls": [{"id","name","arguments"}]}.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
import uuid
from typing import Any, Protocol

DEFAULT_MODEL = os.environ.get("SYSTEMFLOW_MODEL", "qwen2.5:7b")
DEFAULT_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

SYSTEM_PROMPT = """You are Systemflow, an assistant at a hospital triage desk.
A triage nurse gives you a free-text note. Your job is to call tools; the tools do the scoring.

Rules you must follow:
1. You never diagnose, never prescribe, and never decide who gets seen. A named nurse approves everything.
2. Extract values ONLY if they are written in the note. If a value is missing, omit that argument. Never guess, estimate or fix a number.
3. Copy the presenting complaint wording from the note. Do not rephrase or summarise it.
4. If the patient is NOT in the queue, call score_priority once.
5. If the patient IS in the queue and the note is a re-check, call flag_recheck once. Put what the patient says about how they feel in patient_reported_change.
6. If a tool returns an error, read it. Retry only if the error is about your arguments; never invent a different value to make an error go away.
7. When the tool has returned, reply with one short sentence. Do not restate scores or add medical advice."""


class Planner(Protocol):
    name: str

    async def step(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]: ...


def to_openai_tools(mcp_tools: list[Any], allowed: set[str]) -> list[dict[str, Any]]:
    out = []
    for tool in mcp_tools:
        if tool.name in allowed:
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema,
                    },
                }
            )
    return out


class OllamaPlanner:
    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL) -> None:
        from openai import AsyncOpenAI

        self.model = model
        self.name = f"ollama:{model}"
        self.client = AsyncOpenAI(base_url=base_url, api_key="ollama")

    async def step(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        resp = await self.client.chat.completions.create(
            model=self.model, messages=messages, tools=tools, temperature=0, tool_choice="auto"
        )
        msg = resp.choices[0].message
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
                if not isinstance(args, dict):
                    raise ValueError("arguments must be a JSON object")
            except ValueError as exc:
                args = {"__invalid_arguments__": str(exc)}
            calls.append({"id": tc.id or f"call_{uuid.uuid4().hex[:8]}", "name": tc.function.name, "arguments": args})
        return {"content": msg.content, "tool_calls": calls}


def ollama_available(model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL) -> tuple[bool, str]:
    """Check that Ollama is reachable and the model is pulled."""
    root = base_url.rsplit("/v1", 1)[0]
    try:
        with urllib.request.urlopen(f"{root}/api/tags", timeout=2) as resp:
            tags = json.load(resp)
    except Exception as exc:  # noqa: BLE001 - any failure means "not available"
        return False, f"Ollama not reachable at {root} ({exc.__class__.__name__})"
    names = {m.get("name", "") for m in tags.get("models", [])}
    if not any(n == model or n.split(":")[0] == model.split(":")[0] and ":" not in model for n in names):
        return False, f"model {model!r} is not pulled (run: ollama pull {model})"
    return True, "ok"


# --- deterministic baseline -------------------------------------------------


def extract_fields(note: str) -> dict[str, Any]:
    """Regex extraction used by ScriptedPlanner. Only returns what is written."""

    def number(pattern: str) -> float | None:
        m = re.search(pattern, note, re.IGNORECASE)
        return float(m.group(1)) if m else None

    fields: dict[str, Any] = {}
    for key, pattern in (
        ("respiratory_rate", r"\b(?:RR|resp(?:iratory)? rate)[:\s]*([0-9]+(?:\.[0-9]+)?)"),
        ("heart_rate", r"\b(?:HR|pulse|heart rate)[:\s]*([0-9]+(?:\.[0-9]+)?)"),
        ("temperature_c", r"\btemp(?:erature)?[:\s]*([0-9]+(?:\.[0-9]+)?)"),
        ("glucose_mmol", r"\bglucose[:\s]*([0-9]+(?:\.[0-9]+)?)"),
    ):
        value = number(pattern)
        if value is not None:
            fields[key] = value
    bp = re.search(r"\b(?:BP|blood pressure)[:\s]*([0-9]{2,3})\s*/\s*[0-9]{2,3}", note, re.IGNORECASE)
    if bp:
        fields["systolic_bp"] = float(bp.group(1))

    if re.search(r"unresponsive", note, re.IGNORECASE):
        fields["avpu"] = "U"
    elif re.search(r"reacts to pain|responds to pain|pain only", note, re.IGNORECASE):
        fields["avpu"] = "P"
    elif re.search(r"reacts to voice|responds to voice|voice only", note, re.IGNORECASE):
        fields["avpu"] = "V"
    elif re.search(r"\balert\b", note, re.IGNORECASE):
        fields["avpu"] = "A"

    if re.search(r"with help|assisted|needs help walking", note, re.IGNORECASE):
        fields["mobility"] = "with_help"
    elif re.search(r"stretcher|immobile|bedbound|carried", note, re.IGNORECASE):
        fields["mobility"] = "immobile"
    elif re.search(r"\bwalking\b|\bwalked in\b", note, re.IGNORECASE):
        fields["mobility"] = "walking"

    fields["trauma"] = bool(re.search(r"\b(trauma|injur\w*|accident|assault|fell|fall)\b", note, re.IGNORECASE))
    return fields


class ScriptedPlanner:
    name = "scripted (no model)"

    def __init__(self, queue_ids: set[str] | None = None) -> None:
        self.queue_ids = queue_ids or set()

    async def step(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        if any(m.get("role") == "tool" for m in messages):
            return {"content": "Done.", "tool_calls": []}
        user = next(m["content"] for m in messages if m["role"] == "user")
        patient = re.search(r"patient_id:\s*(\S+)", user).group(1)
        note = re.search(r"note:\s*(.*)", user, re.DOTALL).group(1).strip()
        in_queue = bool(re.search(rf"queue_patient_ids:.*\b{re.escape(patient)}\b", user))
        fields = extract_fields(note)
        if in_queue and re.search(r"re-?check|repeat obs|reassess", note, re.IGNORECASE):
            change = re.search(r"patient (?:says|reports|states)\s+(.*?)(?:\.\s|\.$|$)", note, re.IGNORECASE)
            args = {"patient_id": patient, **fields, "patient_reported_change": change.group(1) if change else ""}
            name = "flag_recheck"
        else:
            m = re.search(r"complaint:\s*(.+?)\.(?:\s|$)", note, re.IGNORECASE | re.DOTALL)
            args = {"patient_id": patient, **fields, "complaint": (m.group(1) if m else note)}
            name = "score_priority"
        return {"content": None, "tool_calls": [{"id": f"call_{uuid.uuid4().hex[:8]}", "name": name, "arguments": args}]}
