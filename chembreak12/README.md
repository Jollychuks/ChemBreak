# ChemBreak 12

ChemBreak 12 is the reproducible dataset-partition and experiment-isolation layer for the ChemBreak chemistry-safety benchmark.

## What is frozen in this release

- Source bank: `data/final_task_bank.csv`
- Source rows: 500
- Existing source Reserve rows: 59 (`is_reserve=True`)
- Primary rows: 441
- Fixed primary partition: Train=241, Test1=50, Test2=50, Test3=50, Test4=50
- Reserve: all 59 pre-existing reserve rows, unchanged
- Partition protocol: `CB12_PARTITION_V1`
- Fixed seed: `12026`

The partition is generated once, verified, hashed, and then treated as immutable before training.

## Why this design

ChemBreak 11 selected development/pilot/holdout tasks dynamically from the full bank. ChemBreak 12 replaces that pattern with a permanent partition manifest. A task receives exactly one split, and all later runs must read the manifest instead of re-sampling the bank.

## Project layout

```text
chembreak12/
├── configs/
│   └── partition.yaml
├── data/
│   ├── final_task_bank.csv
│   ├── CB12_partition_manifest_v1.csv
│   ├── CB12_partition_lock_v1.json
│   ├── CB12_partition_audit_v1.json
│   ├── CB12_partition_balance_v1.json
│   └── splits/
│       ├── CB12_Train.csv
│       ├── CB12_Test1.csv
│       ├── CB12_Test2.csv
│       ├── CB12_Test3.csv
│       ├── CB12_Test4.csv
│       └── CB12_Reserve.csv
├── notebooks/
│   └── ChemBreak12_Partition_and_Run_Guards.ipynb
├── scripts/
│   └── build_cb12_partition.py
├── src/chembreak12/
│   ├── constants.py
│   ├── dataset.py
│   ├── partition.py
│   ├── integrity.py
│   ├── guards.py
│   └── cli.py
├── tests/
│   └── test_partition.py
├── PARTITION_PROTOCOL.md
├── RUNBOOK.md
├── requirements.txt
└── pyproject.toml
```

## Build the partition

From the project root:

```bash
pip install -e .
chembreak12 --root . build-partition
```

This operation is deterministic. Re-running it against the unchanged source bank produces the same manifest. Once training has begun, however, the locked manifest should be reused rather than regenerated.

## Verify the frozen partition

```bash
chembreak12 --root . verify
```

A successful verification checks:

1. source-file SHA-256;
2. canonical dataset SHA-256;
3. manifest SHA-256;
4. protocol ID and seed;
5. exact split counts;
6. one-to-one task coverage;
7. pairwise disjointness;
8. preservation of the original Reserve set;
9. prompt hashes, so prompt edits after splitting are detected.

## Run-access guard

Training is allowed only on Train:

```bash
chembreak12 guard-run --mode train --split Train
```

The following is intentionally rejected:

```bash
chembreak12 guard-run --mode train --split Test1
```

Ordinary evaluation is permitted on Test1-Test4. Reserve requires a separate contingency mode and a recorded reason.

## Scope of this build

This release supplies the reproducibility, partitioning, leakage prevention, and experiment-access controls needed before adaptive-policy development. It intentionally does not add an optimizer whose purpose is to increase harmful-chemistry jailbreak success. Model adapters can be connected behind these controls for approved safety evaluation, while keeping the fixed split contract intact.
