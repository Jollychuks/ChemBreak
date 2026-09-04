# ChemBreak3 Output Schema

## Run-level files

| File | Purpose |
| --- | --- |
| `run_manifest.json` | Resolved configuration, seed, software version, task-bank hash, run signature, and verified storage routing. |
| `selected_tasks.csv` | Exact benchmark subset shared by every target and condition. |
| `selected_tasks.summary.json` | HC, HD, OT, and reserve distributions for the selected subset. |
| `state.sqlite3` | Authoritative atomic checkpoint. |
| `checkpoint_snapshot.sqlite3` | Consistent copy suitable for Cloud Storage synchronization. |

## SQLite tables

| Table | Primary key | Contents |
| --- | --- | --- |
| `metadata` | `key` | Run signature and immutable manifest. |
| `assets` | `assignment_id, condition` | Reusable C1 and C2 precomputed assets. |
| `episodes` | `episode_id` | Status, success, query count, reward, and terminal reason. |
| `transcripts` | `episode_id, turn_index` | Raw attack prompt, target response, action, latency, and query error only. |
| `evaluations` | `episode_id, turn_index` | Judge, deterministic verifier, reward, success, and terminal records only. |
| `api_calls` | `id` | Attacker and judge role, model, provider, latency, and token-usage object. |
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

`episode_summary.csv` has one row per episode. `metrics_overall.csv` groups by target and condition. `metrics_by_axis.csv` groups by target, condition, and one ChemHarm axis. `asr_by_query_budget.csv` supports equal-budget curves. `paired_comparisons.csv` contains matched-task exact McNemar results. `transcripts_redacted.csv` removes attack prompts and target responses. `evaluations.csv` contains no raw conversation fields.

The private `transcripts_raw.jsonl` and `evaluations.jsonl` support expert validation and error analysis while preserving a clear data boundary. Raw transcripts are not copied into the release directory unless the configuration explicitly enables raw release.
