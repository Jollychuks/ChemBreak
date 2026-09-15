# ChemBreak 16 — v16.1.1

This release keeps the same fixed 24-task experiment flow and changes the adaptive-MDP representation only.

Key changes:

- hierarchical Q-value decomposition for cross-task reuse;
- smaller taxonomy-free global behavior state;
- separate HC, HD, OT contextual value tables;
- lightweight task-specific residual table;
- weighted averaging of supported components instead of summing duplicate evidence;
- explicit `cold_start` selection label when no learned component supports a decision;
- retained adaptive epsilon and anti-repetition controls;
- live Q diagnostics for every adaptive turn;
- optimized-phase summary remains `null/not_run` until that phase actually runs;
- one Cloud notebook only, stored under `notebooks/`;
- CB16-only data/config/runtime artifacts.

## v16.1.1 audit hardening

- audited every package filename/path and CB16 runtime namespace;
- added an automated prior-version runtime-name/artifact scan;
- moved dependency installation before the pandas-based dataset verification cell for clean-runtime robustness;
- clarified that `CBV15C-*` / `V15C-*` values are immutable source-data identifiers and must not be renamed;
- removed ambiguous wording in the structured-output provider comments.
