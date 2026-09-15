# ChemBreak 16 outputs

The release directory contains:

- `summary.json` — baseline, each learning epoch, optimized metrics, and optimized-minus-baseline ASR delta when available;
- `episodes.csv` — one row per completed episode;
- `turns.csv` — stored prompt/response/judge/decision record per target turn;
- `strategy_summary.csv` — strategy-use counts and mean rewards;
- `policy_diagnostics.csv` — epsilon, hierarchical Q components, support, repetition penalties, blocked actions, and rewards;
- `policy_support_summary.json` — learned-support and non-zero-Q rates by phase/epoch;
- `run_metadata.json` — immutable experiment identity and hashes.

Before the optimized phase has run, optimized ASR and the ASR delta are recorded as `null` rather than as a false zero.
