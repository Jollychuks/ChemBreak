# ChemBreak 12 Fixed Partition Protocol

## 1. Freeze the source bank

`data/final_task_bank.csv` is the sole CB12 source bank. It contains 500 unique `assignment_id` values and 500 unique normalized `benchmark_prompt` values.

Any modification to the source bank changes its SHA-256 and causes the CB12 lock verification to fail.

## 2. Preserve the existing Reserve designation

The source bank already marks 59 rows with `is_reserve=True`. CB12 does not create a new Reserve sample. Those exact 59 task IDs become the permanent Reserve split.

The remaining 441 rows are the only rows eligible for Train/Test assignment.

## 3. Fixed split capacities

The 441 primary tasks are assigned to exact capacities:

- Train: 241
- Test1: 50
- Test2: 50
- Test3: 50
- Test4: 50

The total with Reserve is 500.

## 4. Deterministic task key

Each primary `assignment_id` receives a SHA-256-derived deterministic key using:

```text
protocol = CB12_PARTITION_V1
seed = 12026
```

The key is independent of the physical row order of the input CSV.

## 5. Balanced deterministic assignment

CB12 uses a deterministic greedy allocator with hard split-capacity constraints and soft distribution-balancing objectives.

The allocator balances three benchmark dimensions:

- `hc_id` with weight 2.0
- `hd_id` with weight 2.0
- `ot_id` with weight 3.0

Rare combinations are assigned first. `matrix_id` rarity contributes to ordering so sparse combinations are handled early, but matrix IDs are not treated as indivisible groups: multiple distinct tasks may legitimately share the same matrix cell.

For each task and each split that still has capacity, the allocator computes the incremental normalized squared error between the current category count and the proportional target count. The split with the lowest cost is selected. Cryptographic hashes resolve ties deterministically.

Because capacities are hard constraints, the algorithm finishes with exactly 241/50/50/50/50 primary tasks.

## 6. One task, one split

`assignment_id` is unique in the source bank and unique in the manifest. Every source task appears exactly once in the manifest.

Therefore for any two distinct partitions `S_i` and `S_j`:

```text
S_i ∩ S_j = ∅
```

and the union of all six partitions equals the complete 500-task source bank.

## 7. Prompt-content lock

The normalized text of every `benchmark_prompt` receives a SHA-256 hash. The manifest stores that hash.

If the prompt associated with an existing task ID is edited after partitioning, `verify` fails even when the task ID itself has not changed.

## 8. Partition manifest

`data/CB12_partition_manifest_v1.csv` is the authoritative mapping from task ID to split. It contains:

- `assignment_id`
- `split`
- `protocol_id`
- `seed`
- `partition_key`
- `prompt_sha256`
- `is_reserve`
- `matrix_id`
- `hc_id`
- `hd_id`
- `ot_id`

All downstream code must obtain split membership from this manifest.

## 9. Cryptographic lock

`data/CB12_partition_lock_v1.json` stores:

- source file SHA-256;
- canonical dataset SHA-256;
- manifest SHA-256;
- protocol ID;
- fixed seed;
- exact split counts;
- access rules.

This lock is created before training and must validate before every CB12 run.

## 10. Training/test isolation

CB12 enforces the following data-access contract:

```text
TRAINING  -> Train only
EVALUATION -> Test1, Test2, Test3, Test4
RESERVE -> Reserve only, with a recorded contingency reason
```

Test tasks are not used for policy fitting, reward tuning, hyperparameter selection, prompt-strategy selection, or debugging of the trained policy.

## 11. Reproducibility rule

The partition algorithm may be rerun to demonstrate determinism before training. Once training begins, the existing manifest becomes part of the experiment record and must be reused unchanged.

Any deliberate change to the split requires a new partition protocol/version, not silent regeneration of `CB12_PARTITION_V1`.
