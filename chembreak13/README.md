# ChemBreak 13

ChemBreak 13 is the small-dataset adaptive-MDP experiment discussed for rapid Google Cloud Notebook Enterprise runs.

## Fixed experiment

- Source bank: 500 ChemHarm/ChemBreak tasks, SHA-256 `62df773ce8c4a252bd23350fcd3a8e83fc670864efb0304fe8c65d21d3c7d6ff`.
- Fixed mini dataset: 24 non-reserve tasks selected deterministically and locked in `data/CB13_mini24_manifest_v1.csv`.
- Coverage: all 9 HC categories, all 8 HD categories, and all 15 OT types. Every HD appears exactly 3 times; each HC appears 2 or 3 times.
- Target: ChemDFM (`OpenDFM/ChemDFM-v1.5-8B`).
- Baseline: 24 one-turn original-prompt episodes.
- Learning: 3 epochs × 24 tasks, fresh conversation each epoch, maximum 4 target turns per episode.
- Optimized: 24 tasks with frozen policy, epsilon 0, maximum 4 target turns.
- Total: 120 episodes; maximum 408 target-model queries.

## GitHub → Cloud Notebook Enterprise

1. Put the entire `chembreak13/` folder at the root of your `ChemBreak` GitHub repository and push it to `main`.
2. Open `chembreak13_Cloud_Notebook.ipynb` in Google Cloud Notebook Enterprise.
3. In the first code cell confirm `PROJECT_ID`, `REPO_URL`, `BRANCH`, `PROJECT_SUBDIR`, `EXPERIMENT_REVISION`, and `LIVE`.
4. Run the notebook from top to bottom. It clones/pulls GitHub, verifies the locked 24-task set, installs the CB13-only dependency stack, configures `/content/chembreak13_storage`, runs preflight (including harmless structured-output probes for the actor and judge), loads ChemDFM once, then exposes separate Baseline, Learning, Freeze, Optimized, Results, and Download cells.
5. If a run is interrupted, rerun the notebook and the phase cell. SQLite checkpoints cause completed episodes to be skipped. Do not delete `/content/chembreak13_storage` unless you intentionally want a completely fresh experiment.

See `docs/METHODOLOGY.md` for the pipeline and `docs/OUTPUTS.md` for result files.
