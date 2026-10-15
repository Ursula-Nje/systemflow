"""triage-desk: an MCP server for a nurse-supervised triage desk.

Tools
-----
score_priority   Score vitals + complaint with SATS. Writes a PENDING proposal.
flag_recheck     Compare a re-check against the queued baseline. Writes a
                 PENDING deterioration-flag proposal if the patient looks worse.
submit_to_queue  Apply an APPROVED proposal to the queue. Refuses without a
                 recorded approval from a named clinician (the gate).
get_queue        Read-only view of the queue.

Resource
--------
triage://sats/rules   The scoring tables the tools use.

Run directly:  python -m systemflow.triage_server   (stdio transport)
"""

from __future__ import annotations

import json
import re
from typing import Any

from mcp.server.fastmcp import FastMCP

from . import db, sats

mcp = FastMCP("triage-desk")

WORSENING = re.compile(
    r"\bworse\b|worsening|getting worse|much worse|faint\w*|dizz\w*|can'?t breathe|collaps\w*|confus\w*|drowsy",
    re.IGNORECASE,
)


def _vitals(**kw: Any) -> dict[str, Any]:
    return {k: v for k, v in kw.items() if v is not None}


@mcp.resource("triage://sats/rules")
def sats_rules() -> str:
    """The SATS TEWS tables and discriminator subset the scoring tool uses."""
    return json.dumps(sats.rules_doc(), indent=2)


@mcp.tool()
def score_priority(
    patient_id: str,
    complaint: str,
    respiratory_rate: float | None = None,
    heart_rate: float | None = None,
    systolic_bp: float | None = None,
    temperature_c: float | None = None,
    avpu: str | None = None,
    mobility: str | None = None,
    trauma: bool = False,
    glucose_mmol: float | None = None,
) -> dict[str, Any]:
    """Score a patient's priority with the South African Triage Scale.

    Pass ONLY values stated in the nurse's note; omit anything not recorded
    (never guess). Copy the presenting complaint wording from the note.
    avpu: A/V/P/U (alert, voice, pain, unresponsive). mobility: walking,
    with_help or immobile. Returns a priority with a citation for every point
    awarded. This creates a PENDING proposal; it does NOT change the queue.
    """
    result = sats.score(
        complaint=complaint,
        respiratory_rate=respiratory_rate,
        heart_rate=heart_rate,
        systolic_bp=systolic_bp,
        temperature_c=temperature_c,
        avpu=avpu,
        mobility=mobility,
        trauma=trauma,
        glucose_mmol=glucose_mmol,
    )
    vitals = _vitals(
        respiratory_rate=respiratory_rate,
        heart_rate=heart_rate,
        systolic_bp=systolic_bp,
        temperature_c=temperature_c,
        avpu=sats.normalise_avpu(avpu),
        mobility=sats.normalise_mobility(mobility),
        trauma=trauma,
        glucose_mmol=glucose_mmol,
    )
    payload = {
        "level": result["level"],
        "tews_total": result["tews_total"],
        "basis": result["basis"],
        "citations": result["citations"],
        "warnings": result["warnings"],
        "incomplete": result["incomplete"],
        "complaint": complaint,
        "vitals": vitals,
    }
    proposal_id = db.create_proposal(patient_id, "priority", payload)
    return {
        "proposal_id": proposal_id,
        "status": "PENDING - not in the queue until a named clinician approves",
        "patient_id": patient_id,
        **result,
    }


@mcp.tool()
def flag_recheck(
    patient_id: str,
    respiratory_rate: float | None = None,
    heart_rate: float | None = None,
    systolic_bp: float | None = None,
    temperature_c: float | None = None,
    avpu: str | None = None,
    mobility: str | None = None,
    trauma: bool = False,
    glucose_mmol: float | None = None,
    patient_reported_change: str = "",
) -> dict[str, Any]:
    """Check a waiting patient for possible deterioration (for a second look).

    Compares a re-check against the patient's queued baseline. If they look
    worse (TEWS up by 2+, a higher level, or the patient reports getting
    worse) this creates a PENDING flag proposal. It never changes the queued
    priority and is not a diagnosis. Pass only values stated in the note.
    """
    baseline = db.queue_row(patient_id)
    if baseline is None:
        raise ValueError(f"{patient_id} is not in the queue, so there is no baseline to compare; score_priority first")

    new = sats.score(
        complaint=patient_reported_change,
        respiratory_rate=respiratory_rate,
        heart_rate=heart_rate,
        systolic_bp=systolic_bp,
        temperature_c=temperature_c,
        avpu=avpu,
        mobility=mobility,
        trauma=trauma,
        glucose_mmol=glucose_mmol,
    )
    reasons: list[str] = []
    delta = new["tews_total"] - baseline["last_tews"]
    if delta >= 2:
        reasons.append(f"TEWS rose from {baseline['last_tews']} to {new['tews_total']} (+{delta})")
    if sats.RANK[new["level"]] > sats.RANK[baseline["priority"]]:
        reasons.append(f"Scored level would rise from {baseline['priority']} to {new['level']}")
    quote = WORSENING.search(patient_reported_change or "")
    if quote:
        reasons.append(f"Patient reports a change: matched {quote.group(0)!r} in {patient_reported_change!r}")

    base = {
        "patient_id": patient_id,
        "baseline_priority": baseline["priority"],
        "baseline_tews": baseline["last_tews"],
        "new_tews": new["tews_total"],
        "new_level": new["level"],
        "citations": new["citations"],
        "warnings": new["warnings"],
        "incomplete": new["incomplete"],
    }
    if not reasons:
        return {**base, "flagged": False, "note": "No deterioration signal. No proposal created."}

    vitals = _vitals(
        respiratory_rate=respiratory_rate,
        heart_rate=heart_rate,
        systolic_bp=systolic_bp,
        temperature_c=temperature_c,
        avpu=sats.normalise_avpu(avpu),
        mobility=sats.normalise_mobility(mobility),
        trauma=trauma,
        glucose_mmol=glucose_mmol,
    )
    payload = {**base, "reasons": reasons, "level": new["level"], "vitals": vitals}
    proposal_id = db.create_proposal(patient_id, "recheck_flag", payload)
    return {
        **base,
        "flagged": True,
        "reasons": reasons,
        "proposal_id": proposal_id,
        "status": "PENDING - flag reaches the queue only after a named clinician approves",
        "note": "For a second look, not a verdict. Priority is unchanged until re-scored and approved.",
    }


@mcp.tool()
def submit_to_queue(proposal_id: int) -> dict[str, Any]:
    """Apply an approved proposal to the patient queue.

    The gate: this REFUSES unless a named clinician on the roster has
    approved the proposal. There is no override.
    """
    try:
        return db.commit_proposal(proposal_id)
    except db.GateError as exc:
        raise ValueError(str(exc)) from exc


@mcp.tool()
def get_queue() -> dict[str, Any]:
    """Read-only: the current queue in order, with recheck flags.

    Returns an object, not a bare list: MCP splits a list into one content
    block per item, which makes a one-patient queue look like a single record.
    """
    rows = db.queue_list()
    return {"count": len(rows), "queue": rows}


if __name__ == "__main__":
    db.init()
    mcp.run(transport="stdio")
