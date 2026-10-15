# Systemflow

**An agentic triage assistant that works at the triage desk, not instead of the nurse.**
Built for the African Agentic AI Design Challenge, Health & Wellbeing track ("The Triage Desk").

A nurse types or dictates a note. Systemflow's agent extracts the vitals and complaint, calls
MCP tools that score a **sourced** priority with the South African Triage Scale (SATS), and then
**stops**. Nothing reaches the patient queue until a named nurse or clinician approves. It never
diagnoses, never prescribes and never decides who gets seen.

> **Research prototype. Synthetic data only. Not a medical device.** The SATS tables were
> transcribed from the published scale and must be verified against the official manual before
> any clinical use.

## Run it

```bash
git clone <this repo> && cd systemflow
pip install -r requirements.txt
python -m systemflow.cli demo
```

For the real agent, run a local open-weights model first (patient data never leaves the machine):

```bash
ollama pull qwen2.5:7b      # once
python -m systemflow.cli demo
```

The CLI prints which planner is running. If Ollama is not reachable it says so loudly and falls
back to a **scripted regex baseline with no model**. Use `--planner ollama` to refuse the fallback.

The demo walks through synthetic patients. For each one you play the nurse: type a name from the
roster (`data/clinicians.json`) and approve or reject. There is deliberately no auto-approve flag.

```bash
python -m systemflow.cli run --patient P-100 --note "Complaint: cough. Walking. RR 18, pulse 90, BP 120/80, temp 37.0, alert."
python -m systemflow.cli queue      # the queue, in order
python -m systemflow.cli audit      # every tool call, with inputs, outputs, time and approver
python -m evals.run_evals           # the evaluation suite (see EVALS.md)
```

## What is in the box

| Piece | What it is |
|---|---|
| `systemflow/triage_server.py` | **Our MCP server** (`triage-desk`): `score_priority`, `flag_recheck`, `submit_to_queue` (gated), `get_queue`, plus a `triage://sats/rules` resource |
| `mcp-server-sqlite` | **Borrowed MCP server** (the official SQLite server): stores the audit log, so we did not write a database layer |
| `systemflow/agent.py` | LangGraph workflow: intake, plan + tool calls, verify, human gate, commit |
| `systemflow/planners.py` | Model planner (Ollama, open-weights Qwen) and a scripted no-model baseline |
| `systemflow/sats.py` | Deterministic SATS TEWS scoring. The model never computes a priority |
| `evals/` | 16 checks; results saved per planner mode |

See [ARCHITECTURE.md](ARCHITECTURE.md) for the one-page design and [EVALS.md](EVALS.md) for results,
including a failure that is deliberately not fixed.

## Safety properties (and where they are tested)

- **Every priority cites its source**: each TEWS point and each discriminator names the vital or phrase behind it (E01-E05).
- **Missing observations are never silently low-risk**: they are listed and shown to the nurse (E06).
- **No approval, no queue change**: enforced inside `submit_to_queue` itself, not just in the agent (E08, E09, E10).
- **The model cannot approve**: it is not offered any commit or approval tool (E15). Approval is recorded by the graph's human step, from a roster of named clinicians.
- **Invented numbers are caught**: every number the model passes to a tool must appear in the nurse's note (E11); implausible values are refused (E13).
- **Everything is logged**: inputs, outputs, timestamps and the approving clinician (E14); hostile text in a note is stored as data (E12).

## Licence

MIT. See [LICENSE](LICENSE).
