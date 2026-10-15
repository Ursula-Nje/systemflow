# Systemflow: submission description (Health & Wellbeing, "The Triage Desk")

**Sub-theme:** Symptom-based priority scoring.
**Facility and workflow served:** the triage desk of a public hospital outpatient department, where one or two nurses assess every arriving patient before anyone is assigned a priority.

In busy public outpatient departments, the queue starts at triage. A single desk becomes the bottleneck, and a genuinely urgent patient can sit beside a routine follow-up because nothing prompts a second look until the nurse reaches that card. Systemflow supports the nurse at that desk. It does not replace her.

The nurse enters a free-text note of vitals and presenting complaint. An agent built with LangGraph, using a locally hosted open-weights model (Qwen via Ollama), extracts the values and calls tools on our own MCP server. The server scores a priority with the South African Triage Scale: a Triage Early Warning Score from the vitals, plus clinical discriminators matched in the complaint. Every priority cites the exact vital or phrase behind it. Missing observations are flagged rather than treated as normal. For a patient already waiting, a re-check tool compares new vitals against the baseline and raises a deterioration flag for a second look, never a verdict.

The agent never diagnoses, prescribes or decides who is seen. It pauses at a human-in-the-loop gate, and nothing reaches the queue until a named nurse or clinician on the roster approves. The queue tool enforces this itself, so even a direct call cannot bypass it. A verification step blocks any value the model passes to a tool that does not appear in the nurse's note. Every tool call is logged, with inputs, outputs, timestamps and the approving clinician, through a borrowed official SQLite MCP server.

Patient data stays local, on an open-weights model, and the whole project uses synthetic patients. The repository is MIT licensed, with sixteen evaluation checks, including one documented failure that we deliberately did not fix: keyword matching over-triages negated symptoms such as "no chest pain".
