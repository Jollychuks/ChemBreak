# Output schema

## Coverage states

- `running`: an episode is active.
- `pending_verification`: a target response is saved but an observer or verifier stage is incomplete.
- `ready`: saved turns are complete and the target cell can continue with the next query.
- `complete`: a terminal decision is recorded.

## Episode result fields

`episode_results.csv` includes:

- Run identity: `episode_id`, `assignment_id`, `target_id`, `condition`.
- Coverage: `status`, `evaluated`, `queries_used`, `terminal_reason`.
- Outcome: `verified_success`, `success_label`, `successful_turn`.
- Adaptation: `bootstrap_success`, `adaptive_success`, `adaptive_opportunity`, `adaptive_steps_used`.
- Final state: `final_response_class`, `final_verification_status`, `cumulative_reward`.

## Adaptive metrics

`adaptive_mdp_metrics.csv` includes per-target and all-target rows:

- `overall_asr`
- `bootstrap_asr`
- `conditional_adaptive_asr`
- `bootstrap_failures`
- `post_feedback_adaptive_successes`
- Mean target queries and adaptive steps.

## Audit files

- `action_metrics.csv` counts action use, success, reward, and observed progress.
- `state_action_transitions.csv` records response-class transitions by action.
- `role_call_counts.csv` counts actor, observer, verifier, and adjudicator calls.
- `run_coverage.csv` separates complete and resumable episode states.
- `failures.csv` records historical technical failures without converting them into safety outcomes.

## Privacy

Release transcripts are redacted. Full prompts, target outputs, and raw provider responses are written only under `private/` unless the user explicitly enables raw release output.

