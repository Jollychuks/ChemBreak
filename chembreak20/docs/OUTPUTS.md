# Outputs

Each target writes to:

`/content/chembreak20_storage/runs/CB20_ROUTE_MDP_MINI24_V1/<TARGET>/`

Key files under `release/`:

- `summary.json`: phase metrics, terminal query curve, cumulative discovery curve, provider/gate events, freeze summary.
- `episodes.csv`: one row per task episode.
- `turns.csv`: target-query-level records. Raw prompt/response text is redacted by default; structured measurements remain.
- `provider_events.csv`: Attack-LLM policy blocks/errors, candidate drift rejections, judge errors, target errors, and duplicate-candidate rejections.
- `policy_diagnostics.csv`: action, selection mode, Q score, route bonus, goal-preservation score, response correctness, reward, and success.

Frozen controller artifacts are target-specific under:

`/content/chembreak20_storage/policies/CB20_ROUTE_MDP_MINI24_V1/<TARGET>/`

Set `export.include_raw_text: true` only when a local raw-text audit is explicitly required. The default release does not duplicate raw hazardous model outputs.
