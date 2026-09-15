# ChemBreak 16 changes

- Replaced CB15's duplicated `q_general` + `q_task` structure with hierarchical `q_global`, `q_hc`, `q_hd`, `q_ot`, and `q_task` tables.
- Removed HC/HD/OT identifiers from the global behavioral state so global values can actually transfer across tasks.
- Added a coarser task-state key to improve within-task revisit rates across epochs.
- Combined Q components with a weighted mean over visited components rather than summing duplicated evidence.
- Added explicit `cold_start` mode when no learned component supports a decision.
- Retained gradual epsilon schedule `0.30 → 0.20 → 0.15`, adaptive novelty/negative-feedback bonuses, and anti-repetition controls.
- Restored the fixed CB12 partition firewall: Train24 comes only from `Train`; Holdout12 comes only from `Test1`.
- Added a hard runtime rule preventing any holdout access before policy freeze.
- Added an unseen-task holdout evaluation with no Q updates and no task-specific memory for holdout IDs.
- Fixed summary semantics so phases that have not run report `null` ASR instead of false zero values.
- Added richer hierarchical policy diagnostics and separate train/holdout ASR deltas.
- New namespace/version/storage: `CB16`, `16.0.0`, `/content/chembreak16_storage`.
