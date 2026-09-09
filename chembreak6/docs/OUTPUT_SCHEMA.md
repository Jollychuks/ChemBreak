# ChemBreak6 Output Schema

## Run-level files

| File | Purpose |
| --- | --- |
| `run_manifest.json` | Resolved configuration, seed, software version, task-bank hash, run signature, and verified storage routing. |
| `selected_tasks.csv` | Exact benchmark subset shared by every target and condition. |
| `selected_tasks.summary.json` | HC, HD, OT, and reserve distributions for the selected subset. |
| `state.sqlite3` | Authoritative atomic checkpoint. |
| `checkpoint_snapshot.sqlite3` | Consistent copy suitable for Cloud Storage synchronization. |
| `live_status.json` | Latest compact dashboard state, including totals, ASR, current work, checkpoint, and last verdict. |
| `event_log.jsonl` | Append-only sanitized operational events without attack prompts or target responses. |

## SQLite tables

| Table | Primary key | Contents |
| --- | --- | --- |
| `metadata` | `key` | Run signature and immutable manifest. |
| `assets` | `assignment_id, condition` | Reusable C1 and C2 precomputed assets. |
| `episodes` | `episode_id` | Status, success, query count, reward, and terminal reason. |
| `transcripts` | `episode_id, turn_index` | Raw attack prompt, target response, action, latency, and query error only. |
| `evaluations` | `episode_id, turn_index` | Judge, deterministic verifier, reward, success, and terminal records only. |
| `api_calls` | `id` | Private role diagnostics, including provider, model, latency, usage, finish reason, schema status, validation error, and raw structured response. |
| `failures` | `id` | Stage, exception class, message, and timestamp. |

## Transcript record

```text
episode_id
assignment_id
target_id
condition
turn_index
action_id
action_reason
attack_prompt
target_response
query_attempted
query_error
latency_seconds
```

## Evaluation record

```text
episode_id
turn_index
safety_result
chemistry_result
adjudication_result
deterministic_result
reward
success
terminal_reason
```

## Release tables

`episode_summary.csv` has one row per planned or started episode. `episode_results.csv` adds an evaluated flag, jailbreak success value, `YES`, `NO`, or `NOT_EVALUATED` label, final response class, judge values, chemistry validation, adjudication use, terminal reason, and adaptive-opportunity fields. `success_by_target_condition.csv` calculates ASR from completed evaluations only. `run_coverage.csv` reports `complete`, `pending_judgment`, `ready`, `failed`, and `target_unavailable` states separately. `jailbreak_successes.csv` filters the sanitized episode result table to confirmed successes. `metrics_overall.csv` groups by target and condition. `metrics_by_axis.csv` groups by target, condition, and one ChemHarm axis. `asr_by_query_budget.csv` respects the one-query C0 budget and the five-query C1, C2, and C3 budgets. `paired_comparisons.csv` contains matched-task exact McNemar results. `adaptive_mdp_metrics.csv` separates turn-one C3 successes from post-feedback successes. `release/by_condition` contains an episode and coverage CSV for every condition. `transcripts_redacted.csv` removes attack prompts and target responses. `evaluations.csv` contains no raw conversation fields.

The private `transcripts_raw.jsonl` and `evaluations.jsonl` support expert validation and error analysis while preserving a clear data boundary. A transcript can exist temporarily without a matching evaluation when the target response was saved before an interrupted judge call. Resume processes that saved response first. `successful_attacks_private.csv` contains the first confirmed successful raw turn for controlled expert review. Raw transcripts are not copied into the release directory unless the configuration explicitly enables raw release. The notebook does not display private successful attacks unless `SHOW_PRIVATE_OUTPUTS=True` is set deliberately.
