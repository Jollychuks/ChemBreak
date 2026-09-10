# Output schema

## Coverage states

- `running`: an episode is active.
- `pending_verification`: a response is saved but observation or verification is incomplete.
- `ready`: saved turns are complete and the target cell can continue.
- `complete`: a terminal decision is recorded.

## Results

`episode_results.csv` includes run identity, coverage, outcome, policy breach, successful turn, bootstrap and adaptive success, final response class, verification status, and cumulative reward.

`adaptive_mdp_metrics.csv` includes per-condition and per-target validated ASR, 95% Wilson interval, policy-breach ASR, target-90 status, bootstrap ASR, conditional adaptive ASR, and mean query use.

Each condition has `episode_results_<CONDITION>.csv`, `metrics_<CONDITION>.csv`, and `asr_by_budget_<CONDITION>.csv`.

Audit files include `action_metrics.csv`, `state_action_transitions.csv`, `role_call_counts.csv`, `run_coverage.csv`, `failures.csv`, and `policy_q_values.csv`.

Release transcripts are redacted. Full prompts, outputs, and provider records are stored only under `private/` unless raw release output is explicitly enabled.
