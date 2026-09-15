# Output files

Runtime files are written under `/content/chembreak14_storage/runs/CB14_MDP_MINI24_V1/`.

- `state.sqlite3`: resumable experiment checkpoint. Learning turns and their post-update policy snapshots are committed atomically.
- `release/episodes.csv`: one row per baseline, learning, or optimized episode.
- `release/turns.csv`: one row per target interaction, including selected strategy, Q-values, reward, judge result, prompt and response.
- `release/strategy_summary.csv`: strategy usage and mean reward by phase/epoch.
- `release/summary.json`: Baseline ASR, each learning-epoch ASR, Optimized ASR, and ASR difference.
- `release/run_metadata.json`: experiment identity, target model, task IDs, dataset hashes, policy seed, and run mode.
- `/content/chembreak14_storage/policies/CB14_MDP_MINI24_V1/training_policy.json`: portable mirror of the cumulative learning policy.
- `/content/chembreak14_storage/policies/CB14_MDP_MINI24_V1/frozen_policy.json`: frozen policy used for optimized evaluation.

The notebook's final download ZIP also contains the SQLite checkpoint, runtime YAML, fixed mini-set manifest/lock, and both policy artifacts. Model weights and caches are intentionally excluded.
