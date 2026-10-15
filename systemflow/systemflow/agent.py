"""The Systemflow agent: a LangGraph workflow over two MCP servers.

    intake -> plan (LLM + tool calls) -> verify -> human_gate -> commit

* Our own MCP server ("triage-desk") scores, flags and writes to the queue.
* A borrowed MCP server (the official SQLite server) stores the audit log.
* The LLM decides which tool to call and extracts values from the nurse's
  note. It is never given a tool that can approve or commit anything.
* ``human_gate`` pauses the graph (LangGraph interrupt) until a named
  clinician approves or rejects. Only then does ``commit`` call
  ``submit_to_queue``, which independently re-checks the approval.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import sys
import uuid
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Awaitable, Callable, TypedDict

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from . import db
from .planners import SYSTEM_PROMPT, Planner, to_openai_tools

ROOT = Path(__file__).resolve().parent.parent
LLM_TOOLS = {"score_priority", "flag_recheck", "get_queue"}  # note: no submit_to_queue
NUMERIC_ARGS = ("respiratory_rate", "heart_rate", "systolic_bp", "temperature_c", "glucose_mmol")

AUDIT_DDL = (
    "CREATE TABLE IF NOT EXISTS audit_log("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, actor TEXT NOT NULL, event TEXT NOT NULL,"
    "tool TEXT, patient_id TEXT, inputs TEXT, outputs TEXT, clinician TEXT, ok INTEGER NOT NULL)"
)


class AuditError(RuntimeError):
    pass


def sql_literal(value: Any) -> str:
    """Quote a value for SQLite. The borrowed server takes a raw query string."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("\x00", "").replace("'", "''") + "'"


class Runtime:
    """Owns the two MCP connections, the checkpointer and the planner."""

    def __init__(self, db_path: str, planner: Planner, max_steps: int = 4) -> None:
        self.db_path = db_path
        self.planner = planner
        self.max_steps = max_steps
        self._stack = AsyncExitStack()
        self.triage: ClientSession
        self.sqlite: ClientSession
        self.checkpointer: AsyncSqliteSaver
        self.llm_tools: list[dict[str, Any]] = []

    async def __aenter__(self) -> "Runtime":
        db.init(self.db_path)
        env = {**os.environ, "SYSTEMFLOW_DB": self.db_path, "PYTHONPATH": str(ROOT)}

        own = StdioServerParameters(command=sys.executable, args=["-m", "systemflow.triage_server"], env=env, cwd=str(ROOT))
        read, write = await self._stack.enter_async_context(stdio_client(own))
        self.triage = await self._stack.enter_async_context(ClientSession(read, write))
        await self.triage.initialize()

        borrowed = StdioServerParameters(command="mcp-server-sqlite", args=["--db-path", self.db_path], env=env)
        read, write = await self._stack.enter_async_context(stdio_client(borrowed))
        self.sqlite = await self._stack.enter_async_context(ClientSession(read, write))
        await self.sqlite.initialize()

        self.checkpointer = await self._stack.enter_async_context(
            AsyncSqliteSaver.from_conn_string(self.db_path + ".checkpoints")
        )
        res = await self.sqlite.call_tool("create_table", {"query": AUDIT_DDL})
        if res.isError:
            raise AuditError(f"could not create audit_log: {_text(res)}")
        listed = await self.triage.list_tools()
        self.llm_tools = to_openai_tools(listed.tools, LLM_TOOLS)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self._stack.aclose()

    # -- helpers ------------------------------------------------------------

    async def call_triage(self, name: str, args: dict[str, Any]) -> tuple[bool, str]:
        res = await self.triage.call_tool(name, args)
        return bool(res.isError), _text(res)

    async def audit(
        self,
        *,
        actor: str,
        event: str,
        tool: str | None = None,
        patient_id: str | None = None,
        inputs: Any = None,
        outputs: Any = None,
        clinician: str | None = None,
        ok: bool = True,
    ) -> None:
        """Append to the audit log through the borrowed SQLite MCP server."""
        values = [
            db.now(),
            actor,
            event,
            tool,
            patient_id,
            None if inputs is None else json.dumps(inputs, default=str),
            None if outputs is None else (outputs if isinstance(outputs, str) else json.dumps(outputs, default=str)),
            clinician,
            1 if ok else 0,
        ]
        query = (
            "INSERT INTO audit_log(ts, actor, event, tool, patient_id, inputs, outputs, clinician, ok) VALUES ("
            + ",".join(sql_literal(v) for v in values)
            + ")"
        )
        res = await self.sqlite.call_tool("write_query", {"query": query})
        if res.isError:
            raise AuditError(f"audit write failed: {_text(res)}")

    async def read_audit(self, limit: int = 100) -> list[dict[str, Any]]:
        res = await self.sqlite.call_tool("read_query", {"query": f"SELECT * FROM audit_log ORDER BY id DESC LIMIT {int(limit)}"})
        if res.isError:
            raise AuditError(_text(res))
        return _parse_rows(_text(res))


def _text(res: Any) -> str:
    return "".join(c.text for c in res.content if getattr(c, "type", "") == "text")


def _parse_rows(text: str) -> list[dict[str, Any]]:
    import ast

    return ast.literal_eval(text) if text.strip() else []


def _maybe_json(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return text


# --- graph ------------------------------------------------------------------


class State(TypedDict, total=False):
    patient_id: str
    note: str
    messages: list[dict[str, Any]]
    proposals: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    verification: dict[str, Any]
    decision: dict[str, Any]
    commits: list[dict[str, Any]]
    errors: list[str]
    outcome: str  # committed | rejected | blocked | no_action


def build_graph(rt: Runtime):
    async def intake(state: State) -> dict[str, Any]:
        _, text = await rt.call_triage("get_queue", {})
        queue = _maybe_json(text)
        if not isinstance(queue, dict) or "queue" not in queue:
            raise RuntimeError(f"get_queue returned an unexpected result: {text[:200]}")
        ids = [row["patient_id"] for row in queue["queue"]]
        user = (
            f"patient_id: {state['patient_id']}\n"
            f"queue_patient_ids: {', '.join(ids) if ids else '(none)'}\n"
            f"note: {state['note']}"
        )
        await rt.audit(actor="system", event="intake", patient_id=state["patient_id"], inputs={"note": state["note"]})
        return {
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            "proposals": [],
            "trace": [],
            "errors": [],
        }

    async def plan(state: State) -> dict[str, Any]:
        messages = list(state["messages"])
        proposals: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        errors: list[str] = []
        for _ in range(rt.max_steps):
            out = await rt.planner.step(messages, rt.llm_tools)
            tool_calls = out.get("tool_calls") or []
            assistant: dict[str, Any] = {"role": "assistant", "content": out.get("content")}
            if tool_calls:
                assistant["tool_calls"] = [
                    {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                    for c in tool_calls
                ]
            messages.append(assistant)
            if not tool_calls:
                break
            for call in tool_calls:
                name, args = call["name"], call["arguments"]
                if name not in LLM_TOOLS:
                    is_err, text = True, f"tool {name!r} is not available to the agent"
                elif "__invalid_arguments__" in args:
                    is_err, text = True, f"arguments were not valid JSON: {args['__invalid_arguments__']}"
                else:
                    is_err, text = await rt.call_triage(name, args)
                await rt.audit(
                    actor="agent",
                    event="tool_call",
                    tool=name,
                    patient_id=args.get("patient_id", state["patient_id"]) if isinstance(args, dict) else state["patient_id"],
                    inputs=args,
                    outputs=text,
                    ok=not is_err,
                )
                trace.append({"tool": name, "arguments": args, "is_error": is_err, "output": text})
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": text})
                if is_err:
                    errors.append(f"{name}: {text}")
                    continue
                data = _maybe_json(text)
                if name in ("score_priority", "flag_recheck") and isinstance(data, dict) and "proposal_id" in data:
                    proposals.append(
                        {
                            "proposal_id": data["proposal_id"],
                            "kind": "priority" if name == "score_priority" else "recheck_flag",
                            "patient_id": data["patient_id"],
                            "level": data.get("level") or data.get("new_level"),
                            "tews": data.get("tews_total", data.get("new_tews")),
                            "basis": data.get("basis"),
                            "reasons": data.get("reasons", []),
                            "citations": data.get("citations", []),
                            "warnings": data.get("warnings", []),
                            "incomplete": data.get("incomplete", False),
                        }
                    )
        return {"messages": messages, "proposals": proposals, "trace": trace, "errors": errors}

    async def verify(state: State) -> dict[str, Any]:
        """Block proposals whose inputs are not backed by the nurse's note."""
        note_numbers = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", state["note"])]
        problems: list[str] = []
        for step in state["trace"]:
            if step["is_error"] or step["tool"] not in ("score_priority", "flag_recheck"):
                continue
            args = step["arguments"]
            if args.get("patient_id") != state["patient_id"]:
                problems.append(f"{step['tool']} was called for {args.get('patient_id')!r}, not {state['patient_id']!r}")
            for key in NUMERIC_ARGS:
                value = args.get(key)
                if value is not None and not any(abs(float(value) - n) < 1e-9 for n in note_numbers):
                    problems.append(f"{key}={value} does not appear in the nurse's note")
        ok = not problems
        if not ok:
            for p in state["proposals"]:
                db.set_status(p["proposal_id"], "blocked", rt.db_path)
        await rt.audit(
            actor="system",
            event="verification",
            patient_id=state["patient_id"],
            outputs={"ok": ok, "problems": problems},
            ok=ok,
        )
        update: dict[str, Any] = {"verification": {"ok": ok, "problems": problems}}
        if not ok:
            update["outcome"] = "blocked"
            update["errors"] = state.get("errors", []) + problems
        elif not state["proposals"]:
            update["outcome"] = "blocked" if state.get("errors") else "no_action"
        return update

    def after_verify(state: State) -> str:
        return "human_gate" if state.get("verification", {}).get("ok") and state.get("proposals") else "finish"

    async def human_gate(state: State) -> dict[str, Any]:
        # Everything before interrupt() re-runs on resume, so keep it side-effect free.
        decision = interrupt(
            {
                "patient_id": state["patient_id"],
                "note": state["note"],
                "proposals": state["proposals"],
                "ask": "A named nurse or clinician must approve or reject before anything reaches the queue.",
            }
        )
        clinician = str(decision.get("clinician", ""))
        verdict = str(decision.get("decision", "")).lower()
        note = str(decision.get("note", ""))
        recorded = []
        try:
            for p in state["proposals"]:
                recorded.append(db.record_approval(p["proposal_id"], clinician, verdict, note, rt.db_path))
        except db.GateError as exc:
            await rt.audit(
                actor="human", event="approval_refused", patient_id=state["patient_id"], outputs=str(exc), clinician=clinician, ok=False
            )
            for p in state["proposals"]:
                db.set_status(p["proposal_id"], "blocked", rt.db_path)
            return {"decision": decision, "outcome": "blocked", "errors": state.get("errors", []) + [str(exc)]}
        await rt.audit(
            actor="human",
            event="human_decision",
            patient_id=state["patient_id"],
            inputs={"proposal_ids": [p["proposal_id"] for p in state["proposals"]]},
            outputs={"decision": verdict, "note": note},
            clinician=recorded[0]["clinician"],
        )
        return {"decision": {**decision, "clinician": recorded[0]["clinician"]}, "outcome": "rejected" if verdict == "rejected" else ""}

    def after_gate(state: State) -> str:
        return "commit" if state.get("decision", {}).get("decision") == "approved" and not state.get("outcome") else "finish"

    async def commit(state: State) -> dict[str, Any]:
        commits, errors = [], list(state.get("errors", []))
        for p in state["proposals"]:
            is_err, text = await rt.call_triage("submit_to_queue", {"proposal_id": p["proposal_id"]})
            await rt.audit(
                actor="agent",
                event="commit",
                tool="submit_to_queue",
                patient_id=state["patient_id"],
                inputs={"proposal_id": p["proposal_id"]},
                outputs=text,
                clinician=state["decision"]["clinician"],
                ok=not is_err,
            )
            if is_err:
                errors.append(text)
            else:
                commits.append(_maybe_json(text))
        return {"commits": commits, "errors": errors, "outcome": "blocked" if errors else "committed"}

    async def finish(state: State) -> dict[str, Any]:
        return {"outcome": state.get("outcome") or "no_action"}

    graph = StateGraph(State)
    graph.add_node("intake", intake)
    graph.add_node("plan", plan)
    graph.add_node("verify", verify)
    graph.add_node("human_gate", human_gate)
    graph.add_node("commit", commit)
    graph.add_node("finish", finish)
    graph.add_edge(START, "intake")
    graph.add_edge("intake", "plan")
    graph.add_edge("plan", "verify")
    graph.add_conditional_edges("verify", after_verify, {"human_gate": "human_gate", "finish": "finish"})
    graph.add_conditional_edges("human_gate", after_gate, {"commit": "commit", "finish": "finish"})
    graph.add_edge("commit", "finish")
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=rt.checkpointer)


Approver = Callable[[dict[str, Any]], "dict[str, Any] | Awaitable[dict[str, Any]]"]


async def run_case(rt: Runtime, patient_id: str, note: str, approver: Approver) -> dict[str, Any]:
    """Run one patient note through the graph. ``approver`` plays the nurse."""
    graph = build_graph(rt)
    config = {"configurable": {"thread_id": f"{patient_id}-{uuid.uuid4().hex[:8]}"}}
    result = await graph.ainvoke({"patient_id": patient_id, "note": note}, config)
    gate_payload = None
    interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
    if interrupts:
        gate_payload = interrupts[0].value
        decision = approver(gate_payload)
        if inspect.isawaitable(decision):
            decision = await decision
        result = await graph.ainvoke(Command(resume=decision), config)
    result = dict(result)
    result.pop("__interrupt__", None)
    result["gate_payload"] = gate_payload
    return result
