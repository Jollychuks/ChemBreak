# ChemBreak 14

ChemBreak 14 is the fixed 24-task adaptive-MDP safety-evaluation experiment for Google Cloud Notebook Enterprise.

## Fixed experiment

- Source bank: 500 ChemHarm/ChemBreak tasks, SHA-256 `62df773ce8c4a252bd23350fcd3a8e83fc670864efb0304fe8c65d21d3c7d6ff`.
- Fixed mini dataset: the exact validated 24 non-reserve tasks are frozen and locked in `data/CB14_mini24_manifest_v1.csv`; CB14 does not silently choose a different panel.
- Coverage: all 9 HC categories, all 8 HD categories, and all 15 OT types. Every HD appears exactly 3 times; each HC appears 2 or 3 times.
- Target: ChemDFM (`OpenDFM/ChemDFM-v1.5-8B`).
- Baseline: 24 one-turn original-prompt episodes.
- Learning: 3 epochs × 24 tasks, fresh conversation each epoch, maximum 4 target turns per episode.
- Optimized: 24 tasks with the frozen learned policy, epsilon 0, maximum 4 target turns.
- Total: 120 episodes; maximum 408 target-model queries.

## GitHub → Cloud Notebook Enterprise

1. Put the entire `chembreak14/` folder at the root of your `ChemBreak` GitHub repository and push it to `main`.
2. Open `chembreak14_Cloud_Notebook.ipynb` in Google Cloud Notebook Enterprise.
3. In the first code cell confirm `PROJECT_ID`, `REPO_URL`, `BRANCH`, `PROJECT_SUBDIR`, `EXPERIMENT_REVISION`, and `LIVE`.
4. Run the notebook from top to bottom. It clones/pulls GitHub, verifies the locked 24-task set, installs an isolated pinned CB14 dependency stack, checks for cross-version contamination, configures `/content/chembreak14_storage`, runs preflight, loads ChemDFM once, and then exposes separate Baseline, Learning, Freeze, Optimized, Results, and Download cells.
5. If a run is interrupted, rerun the notebook and the relevant phase cell. SQLite checkpoints skip completed episodes and atomically preserve learning turns with their policy snapshots. If the existing checkpoint does not match the current experiment identity, CB14 refuses to mix the runs.
6. Do not delete `/content/chembreak14_storage` unless you intentionally want a completely fresh CB14 experiment. To start a methodologically distinct run without deleting prior results, change `EXPERIMENT_REVISION`.


## Live notebook display

`LIVE_PROGRESS = True` is enabled by default in the first notebook cell. Baseline, Learning, and Optimized Evaluation print progress while the cell is still running. Each live line reports compact metrics such as assignment ID, turn, abstract action, response class, success, goal progress, reward, running phase ASR, overall progress, target-query count, elapsed time, and ETA. Prompt and response bodies are stored in the result files but are not echoed into the live notebook output.

## Live structured-output reliability

CB14 14.0.0 hardens the Vertex actor/judge adapter. Current Google Gen AI SDK structured output is consumed from `response.parsed` when available, with a validated JSON-text fallback for compatibility. If Vertex returns a truncated result, CB14 retries with a larger output budget and reports the finish reason/token diagnostics rather than accepting malformed judge data. The judge uses `thinking_budget: 0` so its small classification JSON is not crowded out by reasoning tokens.

## Validation scope

The package includes local unit/integration tests, exact dataset/manifest hash checks, package-manifest verification, notebook syntax checks, resume tests, and a complete 120-episode mock run. Live ChemDFM download/inference and live Vertex calls must still be exercised inside your authorized Google Cloud project because those services and credentials are not available during local package validation.

See `docs/METHODOLOGY.md` for the pipeline and interpretation boundary, and `docs/OUTPUTS.md` for result files.
