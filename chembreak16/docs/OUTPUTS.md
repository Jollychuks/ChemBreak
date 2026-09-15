# CB16 outputs

Runtime files are written under `/content/chembreak16_storage/runs/CB16_HIER_MDP_TRAIN24_TEST12_V1/` by default.

The `release/` directory contains `episodes.csv`, `turns.csv`, `summary.json`, `strategy_summary.csv`, `policy_diagnostics.csv`, and `run_metadata.json`. The summary separates Train baseline/learning/optimized metrics from Test1 holdout baseline/optimized metrics and reports train and holdout ASR deltas only when both required phases are complete.

`policy_diagnostics.csv` records selection mode, base/effective epsilon, all five Q components, combined Q, active components, support visits, per-component visit counts, repetition penalty, blocked actions, and reward for every adaptive decision.

Portable policy artifacts are stored under `/content/chembreak16_storage/policies/CB16_HIER_MDP_TRAIN24_TEST12_V1/`. The result ZIP additionally includes the SQLite checkpoint, runtime YAML, CB12 partition provenance, CB16 Train/Holdout manifests and selection lock, and training/frozen policy JSON files.
