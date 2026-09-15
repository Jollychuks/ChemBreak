# Output files

Runtime files are written under `/content/chembreak15_storage/runs/CB15_MDP_MINI24_V1/`.

- `state.sqlite3`: resumable experiment checkpoint. Learning turns and their post-update policy snapshots are committed atomically.
- `release/episodes.csv`: one row per baseline, learning, or optimized episode.
- `release/turns.csv`: one row per target interaction, including selected strategy, state key, Q-values, judge result, decision JSON, reward, prompt, and response.
- `release/policy_diagnostics.csv`: one row per adaptive decision with base/effective epsilon, general Q, task-state Q, combined Q, repetition penalty, adjusted score, blocked actions, visit count, and reward.
- `release/strategy_summary.csv`: strategy usage and mean reward by phase/epoch.
- `release/summary.json`: Baseline ASR, each learning-epoch ASR, Optimized ASR, mean rewards/turns, and ASR difference.
- `release/run_metadata.json`: experiment identity, target model, task IDs, dataset hashes, policy seed, policy schema version, and run mode.
- `/content/chembreak15_storage/policies/CB15_MDP_MINI24_V1/training_policy.json`: portable mirror of the cumulative learning policy.
- `/content/chembreak15_storage/policies/CB15_MDP_MINI24_V1/frozen_policy.json`: frozen policy used for optimized evaluation.

The notebook's final download ZIP also contains the SQLite checkpoint, runtime YAML, fixed mini-set manifest/lock, and both policy artifacts. Model weights and caches are intentionally excluded.
