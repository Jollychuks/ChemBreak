# ChemBreak 15

ChemBreak 15 is the fixed 24-task, state-aware adaptive-MDP safety-evaluation experiment for Google Cloud Notebook Enterprise.

## Fixed experiment

- Source bank: 500 ChemHarm/ChemBreak tasks, SHA-256 `62df773ce8c4a252bd23350fcd3a8e83fc670864efb0304fe8c65d21d3c7d6ff`.
- Fixed mini dataset: the exact validated 24 non-reserve tasks are frozen and locked in `data/CB15_mini24_manifest_v1.csv`; CB15 does not silently choose a different panel.
- Coverage: all 9 HC categories, all 8 HD categories, and all 15 OT types. Every HD appears exactly 3 times; each HC appears 2 or 3 times.
- Target: ChemDFM (`OpenDFM/ChemDFM-v1.5-8B`).
- Baseline: 24 one-turn original-prompt episodes.
- Learning: 3 epochs × 24 tasks, fresh conversation each epoch, maximum 4 target turns per episode.
- Base epsilon: `0.30 → 0.20 → 0.15`; adaptive learning epsilon is capped at `0.35`.
- Optimized: 24 tasks with the frozen learned policy, epsilon `0`, maximum 4 target turns.
- Total: 120 episodes; maximum 408 target-model queries.

## What changed in CB15

CB15 fixes the policy-lock-in failure mode exposed by the preceding development run. Task memory is now `task × state × action` rather than `task × action`; exploration decays more slowly; and repeated non-positive use of the same action receives a penalty and, after two consecutive non-positive repeats, a one-step temporary block when alternatives exist. These rules are generic policy controls; the actor remains limited to abstract, non-operational safety-evaluation strategies.

## GitHub → Cloud Notebook Enterprise

1. Put the entire `chembreak15/` folder at the root of your `ChemBreak` GitHub repository and push it to `main`.
2. Open `chembreak15_Cloud_Notebook.ipynb` in Google Cloud Notebook Enterprise.
3. In the first code cell confirm `PROJECT_ID`, `REPO_URL`, `BRANCH`, `PROJECT_SUBDIR`, `EXPERIMENT_REVISION`, `LIVE`, and `LIVE_PROGRESS`.
4. Run the notebook from top to bottom. It clones/pulls GitHub, verifies the locked 24-task set, installs an isolated pinned CB15 dependency stack, checks cross-version contamination, configures `/content/chembreak15_storage`, runs preflight, loads ChemDFM once, and then exposes separate Baseline, Learning, Freeze, Optimized, Results, and Download cells.
5. If a run is interrupted, rerun the notebook and the relevant phase cell. SQLite checkpoints skip completed episodes and atomically preserve learning turns with their policy snapshots. If the existing checkpoint does not match the current experiment identity, CB15 refuses to mix the runs.
6. Do not delete `/content/chembreak15_storage` unless you intentionally want a completely fresh CB15 experiment. To start a methodologically distinct run without deleting prior results, change `EXPERIMENT_REVISION`.

## Live notebook display

`LIVE_PROGRESS = True` is enabled by default. Baseline, Learning, and Optimized Evaluation display results while their cell is running. Adaptive lines show assignment ID, turn, action, response class, success, goal progress, reward, exploration/exploitation mode, base→effective epsilon, `Qg`, state-specific `Qt`, combined Q, repetition penalty, adjusted score, and blocked actions. Episode-completion lines show running ASR, overall progress, target-query count, elapsed time, and ETA. Prompt and response bodies are stored in the result files but are not echoed into notebook output.

## Live structured-output reliability

CB15 retains the hardened Vertex actor/judge adapter. Current Google Gen AI SDK structured output is consumed from `response.parsed` when available, with a validated JSON-text fallback for compatibility. If Vertex returns a truncated result, CB15 retries with a larger output budget and reports finish-reason/token diagnostics rather than accepting malformed judge data. The judge uses `thinking_budget: 0` so its classification JSON is not crowded out by reasoning tokens.

## Validation scope

The package includes local unit/integration tests, exact dataset/manifest hash checks, package-manifest verification, notebook syntax checks, policy-collapse regression tests, resume tests, and a complete mock run. Live ChemDFM download/inference and live Vertex calls must still be exercised inside your authorized Google Cloud project because those services and credentials are not available during local package validation.

See `docs/METHODOLOGY.md` for the pipeline and interpretation boundary, and `docs/OUTPUTS.md` for result files.
