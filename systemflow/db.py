"""SQLite persistence for proposals, clinician approvals and the triage queue.

Design rule: nothing reaches the queue except through ``commit_proposal``, and
``commit_proposal`` refuses unless ``record_approval`` has stored an approval
from a named clinician on the roster. ``record_approval`` is called by the
human-approval step of the agent graph. It is deliberately NOT exposed as an
MCP tool, so no model can approve its own proposal.

The audit log is a separate table written through the borrowed SQLite MCP
server (see agent.py).
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from . import sats

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = os.environ.get("SYSTEMFLOW_DB", str(ROOT / "systemflow.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS proposals(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('priority','recheck_flag')),
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK(status IN ('pending','committed','rejected','blocked'))
);
CREATE TABLE IF NOT EXISTS approvals(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    proposal_id INTEGER NOT NULL UNIQUE REFERENCES proposals(id),
    clinician TEXT NOT NULL,
    decision TEXT NOT NULL CHECK(decision IN ('approved','rejected')),
    note TEXT,
    decided_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS queue(
    patient_id TEXT PRIMARY KEY,
    priority TEXT NOT NULL,
    last_tews INTEGER NOT NULL,
    last_vitals TEXT NOT NULL,
    recheck_flag INTEGER NOT NULL DEFAULT 0,
    flag_reasons TEXT,
    arrived_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    approved_by TEXT NOT NULL,
    proposal_id INTEGER NOT NULL
);
"""


class GateError(Exception):
    """Raised when an action is refused by the approval gate."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect(path: str | None = None) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(path or DEFAULT_DB, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init(path: str | None = None) -> None:
    with connect(path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)


def load_roster() -> list[dict[str, str]]:
    path = Path(os.environ.get("SYSTEMFLOW_ROSTER", ROOT / "data" / "clinicians.json"))
    return json.loads(path.read_text())["clinicians"]


def roster_name(name: str) -> str | None:
    wanted = " ".join(name.split()).lower()
    for person in load_roster():
        if person["name"].lower() == wanted:
            return person["name"]
    return None


def create_proposal(patient_id: str, kind: str, payload: dict[str, Any], path: str | None = None) -> int:
    with connect(path) as conn:
        cur = conn.execute(
            "INSERT INTO proposals(patient_id, kind, payload, created_at) VALUES (?,?,?,?)",
            (patient_id, kind, json.dumps(payload), now()),
        )
        return int(cur.lastrowid)


def get_proposal(proposal_id: int, path: str | None = None) -> dict[str, Any] | None:
    with connect(path) as conn:
        row = conn.execute("SELECT * FROM proposals WHERE id=?", (proposal_id,)).fetchone()
    if row is None:
        return None
    data = dict(row)
    data["payload"] = json.loads(data["payload"])
    return data


def set_status(proposal_id: int, status: str, path: str | None = None) -> None:
    with connect(path) as conn:
        conn.execute("UPDATE proposals SET status=? WHERE id=? AND status='pending'", (status, proposal_id))


def record_approval(
    proposal_id: int, clinician: str, decision: str, note: str = "", path: str | None = None
) -> dict[str, Any]:
    """Store a named clinician's decision. Called by the human-gate step only."""
    if decision not in ("approved", "rejected"):
        raise GateError("decision must be 'approved' or 'rejected'")
    name = roster_name(clinician or "")
    if name is None:
        raise GateError(f"{clinician!r} is not on the clinician roster; approval not recorded")
    with connect(path) as conn:
        row = conn.execute("SELECT status FROM proposals WHERE id=?", (proposal_id,)).fetchone()
        if row is None:
            raise GateError(f"proposal {proposal_id} does not exist")
        if row["status"] != "pending":
            raise GateError(f"proposal {proposal_id} is already {row['status']}")
        try:
            conn.execute(
                "INSERT INTO approvals(proposal_id, clinician, decision, note, decided_at) VALUES (?,?,?,?,?)",
                (proposal_id, name, decision, note, now()),
            )
        except sqlite3.IntegrityError as exc:
            raise GateError(f"proposal {proposal_id} already has a recorded decision") from exc
        if decision == "rejected":
            conn.execute("UPDATE proposals SET status='rejected' WHERE id=?", (proposal_id,))
    return {"proposal_id": proposal_id, "clinician": name, "decision": decision}


def queue_row(patient_id: str, path: str | None = None) -> dict[str, Any] | None:
    with connect(path) as conn:
        row = conn.execute("SELECT * FROM queue WHERE patient_id=?", (patient_id,)).fetchone()
    return dict(row) if row else None


def queue_list(path: str | None = None) -> list[dict[str, Any]]:
    with connect(path) as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM queue")]
    rows.sort(key=lambda r: (-sats.RANK[r["priority"]], r["arrived_at"], r["patient_id"]))
    for position, row in enumerate(rows, start=1):
        row["position"] = position
        row["last_vitals"] = json.loads(row["last_vitals"])
    return rows


def commit_proposal(proposal_id: int, path: str | None = None) -> dict[str, Any]:
    """Apply an APPROVED proposal to the queue. Refuses without approval."""
    with connect(path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        prop = conn.execute("SELECT * FROM proposals WHERE id=?", (proposal_id,)).fetchone()
        if prop is None:
            raise GateError(f"REFUSED: proposal {proposal_id} does not exist")
        if prop["status"] != "pending":
            raise GateError(f"REFUSED: proposal {proposal_id} is already {prop['status']}")
        appr = conn.execute("SELECT * FROM approvals WHERE proposal_id=?", (proposal_id,)).fetchone()
        if appr is None:
            raise GateError(
                f"REFUSED: no clinician approval is recorded for proposal {proposal_id}. "
                "The queue only changes after a named nurse or clinician approves."
            )
        if appr["decision"] != "approved":
            raise GateError(f"REFUSED: proposal {proposal_id} was rejected by {appr['clinician']}")

        payload = json.loads(prop["payload"])
        stamp = now()
        patient_id = prop["patient_id"]
        existing = conn.execute("SELECT * FROM queue WHERE patient_id=?", (patient_id,)).fetchone()

        if prop["kind"] == "priority":
            vitals = json.dumps(payload["vitals"])
            if existing is None:
                conn.execute(
                    "INSERT INTO queue(patient_id, priority, last_tews, last_vitals, recheck_flag, flag_reasons,"
                    " arrived_at, updated_at, approved_by, proposal_id) VALUES (?,?,?,?,0,NULL,?,?,?,?)",
                    (patient_id, payload["level"], payload["tews_total"], vitals, stamp, stamp, appr["clinician"], proposal_id),
                )
            else:
                conn.execute(
                    "UPDATE queue SET priority=?, last_tews=?, last_vitals=?, recheck_flag=0, flag_reasons=NULL,"
                    " updated_at=?, approved_by=?, proposal_id=? WHERE patient_id=?",
                    (payload["level"], payload["tews_total"], vitals, stamp, appr["clinician"], proposal_id, patient_id),
                )
            effect = f"{patient_id} placed in queue as {payload['level']}"
        else:  # recheck_flag
            if existing is None:
                raise GateError(f"REFUSED: {patient_id} is not in the queue, so there is nothing to flag")
            conn.execute(
                "UPDATE queue SET recheck_flag=1, flag_reasons=?, updated_at=?, approved_by=?, proposal_id=?"
                " WHERE patient_id=?",
                (json.dumps(payload["reasons"]), stamp, appr["clinician"], proposal_id, patient_id),
            )
            effect = f"{patient_id} flagged for a second look (priority unchanged: {existing['priority']})"

        conn.execute("UPDATE proposals SET status='committed' WHERE id=?", (proposal_id,))
    return {"proposal_id": proposal_id, "committed": True, "approved_by": appr["clinician"], "effect": effect}
