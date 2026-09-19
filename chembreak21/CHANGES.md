# ChemBreak21 v21.0.0

- Replaced the previous 24-task bank with the locked 28-row uploaded prompt dataset.
- Attack LLM changed to Gemini 3.1 Pro Preview.
- Judge locked to Gemini 3.8 Flash.
- Removed HC/HD/OT Q layers because the new dataset has no informative variation on those labels.
- Added exact successful trajectory memory and cross-epoch replay.
- Added replay-then-recovery semantics under the fixed four-turn learning-epoch budget.
- Added confirmed-route evidence across distinct epochs.
- Final stage now executes entire ranked successful routes in fresh target conversations before synthesized fallbacks.
- Fixed the all-actions-blocked bug: provider-blocked actions are never silently re-enabled.
- Reward now uses correctness/fidelity-adjusted progress.
- Added immutable raw target-query auditing before judge evaluation.
- Public release artifacts exclude the raw SQLite checkpoint and exact route prompt text.
