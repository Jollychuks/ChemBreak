# ChemBreak 12 — Adaptive MDP safety evaluation

ChemBreak 12 is a direct engineering successor to ChemBreak 11. It keeps the familiar Google Cloud Notebook Enterprise + GitHub + `/content` storage workflow and replaces the old development/pilot/full-bank sampling scheme with a fixed partition created before training.

## Fixed benchmark
- Train: 241
- Test1: 50
- Test2: 50
- Test3: 50
- Test4: 50
- Reserve: 59 (preserves the source `is_reserve=True` rows)

## Active conversational target models
- ChemDFM — `OpenDFM/ChemDFM-v1.5-8B`
- ChemLLM — `AI4Chem/ChemLLM-7B-Chat-1_5-SFT`

## Cloud workflow
1. Put the `chembreak12/` folder in the configured GitHub repository.
2. Open `notebooks/chembreak12_Cloud_Notebook.ipynb` in Google Cloud Notebook Enterprise.
3. In Cell 1 confirm `PROJECT_ID`, `REPO_URL`, `BRANCH`, `PROJECT_SUBDIR`, `PHASE`, `EXPERIMENT_REVISION`, and `LIVE`.
4. Run top to bottom.
5. Use `PHASE='train'` first. Freeze the policy only after both target runs complete.
6. Switch to `test1`–`test4` only after a frozen policy exists.

The notebook never runtime-resamples the benchmark. Every run verifies `data/CB12_partition_manifest_v1.csv` and its lock.
