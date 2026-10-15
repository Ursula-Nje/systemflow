"""Command line interface.

    python -m systemflow.cli demo                 # walk through synthetic patients
    python -m systemflow.cli run --patient P-100 --note "Complaint: ... RR 18, ..."
    python -m systemflow.cli queue
    python -m systemflow.cli audit

The nurse is asked to type their name and approve or reject in the terminal.
There is no flag that approves automatically; that is the point of the gate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

from . import db
from .agent import Runtime, run_case
from .planners import DEFAULT_MODEL, OllamaPlanner, Planner, ScriptedPlanner, ollama_available

DATA = db.ROOT / "data" / "synthetic_patients.json"
BAR = "=" * 72


def choose_planner(kind: str) -> Planner:
    if kind == "scripted":
        print("[planner] scripted regex baseline: NO MODEL is running.")
        return ScriptedPlanner()
    ok, why = ollama_available()
    if ok:
        print(f"[planner] open-weights model via Ollama: {DEFAULT_MODEL}")
        return OllamaPlanner()
    if kind == "ollama":
        sys.exit(f"[planner] {why}. Start Ollama and run `ollama pull {DEFAULT_MODEL}`, or use --planner scripted.")
    print(f"[planner] {why}.")
    print("[planner] FALLING BACK to the scripted regex baseline: NO MODEL is running, so this is not a model-driven run.")
    return ScriptedPlanner()


def print_proposal(payload: dict[str, Any]) -> None:
    print(f"\n{BAR}\nAWAITING NURSE APPROVAL  patient {payload['patient_id']}\n{BAR}")
    print(f"Note: {payload['note']}\n")
    for p in payload["proposals"]:
        if p["kind"] == "priority":
            print(f"Proposed priority: {p['level']}   (TEWS {p['tews']}, basis: {p['basis']})")
        else:
            print(f"Proposed deterioration FLAG (for a second look; priority unchanged). New TEWS {p['tews']}")
            for reason in p["reasons"]:
                print(f"  reason: {reason}")
        for line in p["citations"]:
            print(f"  - {line}")
        for warning in p["warnings"]:
            print(f"  !! {warning}")
    print("\nThe agent does not diagnose, prescribe or decide who is seen. Nothing reaches the queue until you approve.")


def human_approver(payload: dict[str, Any]) -> dict[str, Any]:
    print_proposal(payload)
    names = ", ".join(p["name"] for p in db.load_roster())
    while True:
        clinician = input(f"Clinician name (roster: {names}): ").strip()
        if db.roster_name(clinician):
            break
        print("  Not on the roster. Try again.")
    while True:
        answer = input("Approve? [y = approve / n = reject]: ").strip().lower()
        if answer in ("y", "n"):
            break
    note = input("Optional note: ").strip()
    return {"decision": "approved" if answer == "y" else "rejected", "clinician": clinician, "note": note}


def show_result(result: dict[str, Any]) -> None:
    print(f"\nOutcome: {result['outcome'].upper()}")
    for commit in result.get("commits", []):
        print(f"  {commit['effect']} (approved by {commit['approved_by']})")
    for err in result.get("errors", []):
        print(f"  ! {err}")


def show_queue(path: str) -> None:
    rows = db.queue_list(path)
    print(f"\n{BAR}\nQUEUE\n{BAR}")
    if not rows:
        print("(empty)")
    for r in rows:
        flag = "  [RE-CHECK FLAG: " + "; ".join(json.loads(r["flag_reasons"])) + "]" if r["recheck_flag"] else ""
        print(f"{r['position']}. {r['patient_id']:<8} {r['priority']:<7} TEWS {r['last_tews']}  approved by {r['approved_by']}{flag}")


async def cmd_run(args: argparse.Namespace) -> None:
    planner = choose_planner(args.planner)
    async with Runtime(args.db, planner) as rt:
        result = await run_case(rt, args.patient, args.note, human_approver)
        show_result(result)
        show_queue(args.db)


async def cmd_demo(args: argparse.Namespace) -> None:
    data = json.loads(DATA.read_text())["patients"]
    skip = {"sql_injection_in_note", "implausible_vital"}
    cases = [p for p in data if p["tag"] not in skip]
    if args.case:
        cases = [p for p in data if p["id"] == args.case]
        if not cases:
            sys.exit(f"unknown case {args.case}; choose from {[p['id'] for p in data]}")
    planner = choose_planner(args.planner)
    async with Runtime(args.db, planner) as rt:
        for case in cases:
            patient_id = case.get("patient_id", case["id"])
            result = await run_case(rt, patient_id, case["note"], human_approver)
            show_result(result)
        show_queue(args.db)


async def cmd_queue(args: argparse.Namespace) -> None:
    db.init(args.db)
    show_queue(args.db)


async def cmd_audit(args: argparse.Namespace) -> None:
    async with Runtime(args.db, ScriptedPlanner()) as rt:
        for row in reversed(await rt.read_audit(args.limit)):
            who = f" by {row['clinician']}" if row["clinician"] else ""
            print(f"{row['ts']}  {row['actor']:<6} {row['event']:<14} {row['tool'] or '':<16} {row['patient_id'] or '':<8}{who}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="systemflow", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=os.environ.get("SYSTEMFLOW_DB", db.DEFAULT_DB))
    parser.add_argument("--planner", choices=["auto", "ollama", "scripted"], default="auto")
    sub = parser.add_subparsers(dest="cmd", required=True)
    demo = sub.add_parser("demo", help="walk through the synthetic patients")
    demo.add_argument("--case", help="run a single synthetic case id, e.g. P-004")
    run = sub.add_parser("run", help="triage one note")
    run.add_argument("--patient", required=True)
    run.add_argument("--note", required=True)
    sub.add_parser("queue", help="show the queue")
    audit = sub.add_parser("audit", help="show the audit log")
    audit.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    handlers = {"demo": cmd_demo, "run": cmd_run, "queue": cmd_queue, "audit": cmd_audit}
    asyncio.run(handlers[args.cmd](args))


if __name__ == "__main__":
    main()
