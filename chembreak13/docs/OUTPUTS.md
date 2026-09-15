# Output files

Runtime files are written under `/content/chembreak13_storage/runs/CB13_MDP_MINI24_V1/`.

- `state.sqlite3`: resumable experiment checkpoint.
- `release/episodes.csv`: one row per baseline, learning, or optimized episode.
- `release/turns.csv`: one row per target interaction, including selected strategy, Q-values, reward, judge result, prompt and response.
- `release/strategy_summary.csv`: strategy usage and mean reward by phase/epoch.
- `release/summary.json`: Baseline ASR, each learning-epoch ASR, Optimized ASR, and ASR difference.
- `/content/chembreak13_storage/policies/CB13_MDP_MINI24_V1/training_policy.json`: cumulative learning policy.
- `/content/chembreak13_storage/policies/CB13_MDP_MINI24_V1/frozen_policy.json`: frozen policy used for optimized attack.
