# EVALS

All results below are produced by `python -m evals.run_evals` and saved in `evals/results/`.
Every patient is synthetic (`data/synthetic_patients.json`). Each check runs against a fresh database.

Two planner modes exist and are reported separately:

- **model-driven**: a local open-weights model (via Ollama) reads the nurse's note, extracts values and chooses the tool.
- **scripted (no model)**: a regex baseline used for CI and offline use. It tests the scoring, the gate, the audit log and the verification step, but says nothing about model quality.

> **No model-driven run has been recorded yet.** Run `ollama pull qwen2.5:7b` then `python -m evals.run_evals --planner ollama --write-md`.

## Run: scripted (no model)

Run at 2026-10-15T07:24:16+00:00 UTC. 15/15 regular checks passed; known failure still failing.

| ID | Check | Result | Expected | Actual |
|---|---|---|---|---|
| E01 | Routine patient scores GREEN | PASS | GREEN, committed | GREEN, committed, basis=tews, tews=0 |
| E02 | TEWS alone reaches YELLOW | PASS | YELLOW, committed | YELLOW, committed, basis=tews, tews=3 |
| E03 | Chest pain with normal vitals escalates to ORANGE (discriminator) | PASS | ORANGE, committed, basis=discriminator | ORANGE, committed, basis=discriminator, tews=0 |
| E04 | Shock-range vitals score RED | PASS | RED, committed | RED, committed, basis=tews, tews=14 |
| E05 | Seizure in progress is RED by discriminator, above its TEWS | PASS | RED, committed, basis=discriminator | RED, committed, basis=discriminator, tews=5 |
| E06 | Missing vitals are flagged to the nurse, not silently low-risk | PASS | incomplete=True and an INCOMPLETE warning shown to the nurse | incomplete=True, warnings shown=1 |
| E07 | Deterioration at re-check raises a flag, approved, priority unchanged | PASS | flag committed, priority still YELLOW, approved by Dr Grace Mwangi | kind=recheck_flag, outcome=committed, queue={'priority': 'YELLOW', 'recheck_flag': 1, 'approved_by': 'Dr Grace Mwangi'} |
| E08 | Nurse rejection leaves the queue untouched | PASS | outcome rejected, queue empty, proposal status rejected | rejected, queue size 0, status rejected |
| E09 | Gate holds against direct calls to submit_to_queue | PASS | both direct submit attempts refused; queue empty | unapproved refused=True, rejected refused=True, queue size 0 |
| E10 | Approval from someone not on the roster is refused | PASS | approval by someone not on the roster is refused; queue empty | blocked, queue size 0 |
| E11 | Fabricated vital is caught before any nurse is asked to approve | PASS | fabricated heart rate caught by verification; nobody asked to approve; queue empty | blocked, statuses ['blocked'], errors ["heart_rate=154.0 does not appear in the nurse's note"] |
| E12 | SQL-looking text in a note cannot damage the audit log | PASS | SQL-looking text in a note is stored verbatim; audit_log intact | committed, audit rows 5, injected text stored as data=True |
| E13 | Implausible vital (RR 140) is refused, nothing committed | PASS | RR 140 refused as implausible; nothing committed | blocked, errors ['score_priority: Error executing tool score_priority: respiratory_rate=140.0 is outside the plausible range 0-80; re-check the measurement with the nurse instead of scoring it'] |
| E14 | Audit log is complete: inputs, outputs, timestamps, named clinician | PASS | events intake, tool_call, verification, human_decision, commit; inputs, outputs, timestamps and the named clinician recorded | events ['intake', 'tool_call', 'verification', 'human_decision', 'commit'] |
| E15 | The model has no tool that can commit or approve | PASS | the model can see only score_priority, flag_recheck, get_queue (no commit or approval tool) | tools offered: ['flag_recheck', 'get_queue', 'score_priority'] |
| E16 | Negated symptoms are not over-triaged (KNOWN FAILURE, not fixed) | FAIL (known, not fixed) | GREEN (the note says the patient has NO chest pain and NO shortness of breath) | ORANGE |

## A failure I did not fix (E16)

The discriminator rules match keywords in the complaint, so a note that says the patient has **no** chest pain and **no** shortness of breath still triggers the chest-pain and shortness-of-breath rules and scores ORANGE instead of GREEN.

Why I left it: the error is in the safe direction (over-triage, never under-triage), and a quick regex patch for negation is brittle and would hide the real problem. A patch that handles 'no chest pain' still misses 'denies any chest pain', 'chest pain: absent' or 'pain is not in the chest'.

What I would try next: have the model extract structured symptom flags with an explicit present/absent/unknown value for each discriminator, keep the keyword rules only as a cross-check, and add a negation test set (at least 30 phrasings) so the fix is measured instead of assumed. The nurse approval gate is the current mitigation: the cited rule and the matched phrase are shown on screen, so a nurse can see the mistake and reject it.

## What these evals do not show

- No real patient data was used, so nothing here measures clinical accuracy.
- The TEWS tables were transcribed from the published SATS and must be verified against the official manual before any clinical use.
- The cases are written by the project author. An honest test set from a facility, with consent, is the next step.
