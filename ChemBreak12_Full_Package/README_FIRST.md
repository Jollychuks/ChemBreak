# ChemBreak 12 — Full Google Cloud Notebook Enterprise Package

This is the clean ChemBreak 12 package to use going forward.

## Run only this notebook

`ChemBreak12_Adaptive_MDP_Cloud.ipynb`

The earlier `Partition_and_Run_Guards` and `Cloud_Enterprise_Fixed` notebooks are not required.

## Google Cloud Notebook Enterprise workflow

1. Upload this ZIP to your Google Cloud Notebook Enterprise environment.
2. Extract the ZIP.
3. Open `ChemBreak12_Adaptive_MDP_Cloud.ipynb`.
4. Run the notebook from Cell 1 downward.

The notebook is standalone: it contains an embedded byte-for-byte copy of the source task bank, while `data/final_task_bank.csv` is also included separately for transparency and audit.

## First run

On the first run the notebook:

- validates the 500-task source bank;
- preserves the 59 existing Reserve tasks;
- creates the fixed CB12 partition exactly once;
- assigns Train=241, Test1=50, Test2=50, Test3=50, Test4=50, Reserve=59;
- writes the partition manifest, split CSVs, hashes, and lock files into the CB12 workspace;
- checks pairwise disjointness and complete task coverage;
- records the fixed protocol (`CB12_PARTITION_V1`) and seed (`12026`);
- configures the Google Cloud runtime and CB12-only cache/storage paths;
- defines the active multi-turn targets ChemDFM and ChemLLM;
- enforces Train-only access while mode is `train`.

## Later runs

The notebook does not repartition. It verifies the existing lock and reuses the exact same manifest. If the source, prompt hashes, manifest, protocol, or seed no longer match, verification stops rather than silently creating a new experimental split after training has started.

## Active target models

- ChemDFM: `OpenDFM/ChemDFM-v1.5-8B`
- ChemLLM: `AI4Chem/ChemLLM-7B-Chat-1_5-SFT`

Models are loaded one at a time by default to reduce GPU-memory pressure.

## Package contents

```text
ChemBreak12_Full_Package/
├── ChemBreak12_Adaptive_MDP_Cloud.ipynb   <- notebook to run
├── README_FIRST.md
├── requirements.txt
├── SHA256SUMS.txt
├── config/
│   └── experiment_config.json
├── data/
│   └── final_task_bank.csv
└── docs/
    └── PARTITION_PROTOCOL.md
```

## Important storage behavior

The notebook creates its runtime workspace on Google Cloud when executed. The partition manifest and experiment state therefore come from the notebook's first run in that workspace, not from a pre-generated manifest hidden inside this ZIP.

For reproducible continuation across sessions, keep the CB12 workspace/storage that the notebook creates rather than deleting the lock and split files.

## Scope

This package contains the frozen-dataset protocol, split creation/verification, train/evaluation firewall, Google Cloud preflight, target-model registry and loader, abstract adaptive-MDP configuration, run records, and result schemas. Chemistry-specific harmful prompt tactics are not embedded in the package.
