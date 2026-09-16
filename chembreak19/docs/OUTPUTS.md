# ChemBreak19 outputs

The release directory contains:

- `summary.json` — primary ASR, ASR@1, ASR@4, target-evaluable ASR, recovery rate, success retention, trajectory coverage, provider-block statistics, and baseline-to-optimized delta;
- `baseline_audit.csv` — exact Baseline prompts/responses plus judge scores and gated success;
- `episodes.csv` — one row per episode;
- `turns.csv` — per-target-turn action, realization, source, prompt, response, judge data, policy diagnostics, reward, and latency;
- `provider_events.csv` — Attack-LLM provider policy blocks, separate from target turns;
- `strategy_summary.csv` — action/source usage and mean rewards;
- `policy_diagnostics.csv` — epsilon, hierarchical Q values, support-confidence values, trajectory replay metadata, blocked actions, and rewards;
- `policy_support_summary.json` — learned-support and non-zero-Q rates by phase/epoch;
- `candidate_evidence_memory.csv` — candidate-level diagnostic evidence;
- `trajectory_memory.csv` — accumulated successful-trajectory evidence and outcomes;
- `trajectory_rankings.csv` — frozen successful-trajectory ranking per task;
- `memory_summary.json` — candidate and trajectory coverage plus freeze summary;
- `run_metadata.json` — experiment identity, model roles, settings, and source hashes.

The policy artifact directory contains `training_policy.json`, `frozen_policy.json`, `training_evidence.json`, `frozen_evidence.json`, `training_trajectories.json`, `frozen_trajectories.json`, `frozen_trajectory_rankings.json`, and `freeze_snapshot.json`.
