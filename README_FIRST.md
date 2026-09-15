# Start here — ChemBreak 14 (14.0.0)

Use the `chembreak14/` folder in this package. Add that folder to the root of your GitHub repository `https://github.com/Jollychuks/ChemBreak.git`, commit, and push to `main`.

Then open `chembreak14/chembreak14_Cloud_Notebook.ipynb` (or the identical copy under `chembreak14/notebooks/`) in Google Cloud Notebook Enterprise and run the cells from top to bottom.

The first notebook code cell contains the familiar editable values: `PROJECT_ID`, `REPO_URL`, `BRANCH`, `PROJECT_SUBDIR`, `EXPERIMENT_REVISION`, `LIVE`, and `LIVE_PROGRESS`.

CB14 uses only `/content/chembreak14_storage`. It does not reuse previous-version checkpoints, policies, caches, or runtime files. A checkpoint whose experiment identity does not match the current CB14 method is rejected rather than silently reused.

## Live notebook output

`LIVE_PROGRESS = True` by default. While Baseline, each Learning epoch, and Optimized Evaluation are running, the active notebook cell prints progress immediately. You will see the assignment ID, turn number, abstract action ID, response class, success status, goal-progress score, reward, running phase ASR, overall completed episodes, target-query count, elapsed time, and ETA. Prompt and response text are not echoed into the notebook output.

At the end of each phase, the cell also prints the full phase summary JSON. The final Results cell exports the complete CSV/JSON result set.

## Important

CB14 carries forward the exact validated 24-task mini panel from the preceding build rather than silently selecting a different set merely because the package version changed. The panel membership, source-bank hash, manifest hash, taxonomy coverage, and assignment-ID hash are all locked and checked during preflight.
