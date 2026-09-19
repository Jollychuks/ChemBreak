# ChemBreak21 outputs

For each target under the run directory:

- `state.sqlite3` — **internal raw audit only**, including exact prompts/responses and all target-query records.
- `release/episodes.csv` — phase/epoch task outcomes.
- `release/turns.csv` — judged turns with raw prompt/response text redacted by default.
- `release/target_queries.csv` — every actual target query, including judge-error queries; text redacted by default.
- `release/provider_events.csv` — provider, candidate-gate and technical events.
- `release/route_rankings_public.json` — route IDs, abstract actions and reliability metadata; exact route prompt text excluded.
- `release/summary.json` — ASR, conditional-evaluable ASR, query counts, cumulative discovery curve, replay/recovery counts and freeze coverage.

Internal policy artifacts use filenames containing `_INTERNAL` where exact trajectory prompt text may be present. Do not publish those files without separate review.
