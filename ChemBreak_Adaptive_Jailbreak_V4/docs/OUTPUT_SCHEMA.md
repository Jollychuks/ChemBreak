# ChemBreak V4 Output Schema

## Run-level files

| File | Purpose |
| --- | --- |
| `run_manifest.json` | Resolved configuration, seed, software version, task-bank hash, and run signature. |
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
| `turns` | `episode_id, turn_index` | Complete prompt, response, judge, deterministic, reward, and error record. |
| `api_calls` | `id` | Attacker and judge role, model, provider, latency, and token-usage object. |
| `failures` | `id` | Stage, exception class, message, and timestamp. |

## Turn record

Each turn JSON object contains:

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
safety_result
chemistry_result
adjudication_result
deterministic_result
reward
success
terminal_reason
query_attempted
query_error
latency_seconds
```

## Release tables

`episode_summary.csv` has one row per episode. `metrics_overall.csv` groups by target and condition. `metrics_by_axis.csv` groups by target, condition, and one ChemHarm axis. `asr_by_query_budget.csv` supports equal-budget curves. `paired_comparisons.csv` contains matched-task exact McNemar results. `turns_redacted.csv` removes attack prompts and target responses.

The private `turns_raw.jsonl` is required for expert validation and error analysis. It is not copied into the release directory unless the configuration explicitly enables raw release.

