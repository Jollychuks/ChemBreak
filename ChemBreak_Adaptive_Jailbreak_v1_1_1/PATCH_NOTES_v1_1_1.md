# ChemBreak Adaptive Jailbreak v1.1.1 patch notes

This patch repairs the frozen-asset preparation failures observed in the v1.1 TEST run while preserving the earlier v1.1 package and GCS outputs.

## Fixed

- Repeated single-turn assets now receive structural validation retries when the attacker returns fewer or more than the configured 5 attempts.
- A top-level JSON list is accepted only when it can be unambiguously wrapped as `routes`, `attempts`, or `queries` for the current stage.
- Graph, repeated-single, and fixed-multi assets each receive up to 3 total model generations by default.
- Exact structural defects are fed back on retry without silently changing the benchmark task.
- C2 fixed-multi prompt schema now shows the configured number of query placeholders instead of a hard-coded three-query example.
- `prepare` now writes `public/prepare_summary.json` and exits nonzero when selected tasks still lack complete frozen assets.
- Completed assets remain resumable and are skipped on rerun.

## v1.1 compatibility bridge

`scripts/import_previous_assets.py` can copy only `status=complete` frozen assets from a compatible v1.1 GCS run into the new v1.1.1 namespace. The v1.1 source is read-only and is never modified.

For the earlier 8-task TEST run, this means the five completed v1.1 assets can be reused and only the three missing task assets need regeneration.

## Validation performed during packaging

- Python compilation passed.
- Offline unit tests passed: 9/9.
- Notebook JSON parsed successfully.
- Version and folder-name audit found no unintended v1.1 naming leftovers. The only v1.1 references are the deliberate compatibility-import source references.
- No live target-model or jailbreak calls were made while producing this patch.
