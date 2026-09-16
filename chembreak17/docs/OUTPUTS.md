# ChemBreak17 outputs

The release directory contains:

- `summary.json` — ASR, ASR@1, ASR@4, recovery rate, success retention, evidence coverage, and baseline-to-optimized delta;
- `episodes.csv` — one row per episode;
- `turns.csv` — per-turn action, candidate ID, realization ID, source, prompt, response, judge data, policy diagnostics, reward, and latency;
- `strategy_summary.csv` — action/source usage counts and mean rewards;
- `policy_diagnostics.csv` — epsilon, hierarchical Q components, support, blocked actions, candidate source, and evidence rank when used;
- `policy_support_summary.json` — support and non-zero-Q rates by phase/epoch;
- `evidence_memory.csv` — accumulated exact-candidate evidence;
- `candidate_rankings.csv` — frozen successful-candidate ranking per task;
- `evidence_summary.json` — training and freeze evidence coverage;
- `run_metadata.json` — immutable experiment identity, model roles, settings, and source hashes.

The policy artifact directory additionally contains `training_policy.json`, `frozen_policy.json`, `training_evidence.json`, `frozen_evidence.json`, `frozen_rankings.json`, and `freeze_snapshot.json`.
