# ChemBreak 16

ChemBreak 16 is a fixed-24-task adaptive-MDP safety-evaluation package for Google Cloud Notebook Enterprise.

## Experiment flow

The experiment deliberately keeps one fixed 24-task panel throughout:

`Baseline → Learning Epoch 1 → Learning Epoch 2 → Learning Epoch 3 → Freeze Policy → Optimized Evaluation → Results`

There is no train/test/holdout split inside this package. The purpose of this version is to change the MDP value architecture while keeping the experiment panel and phase structure stable.

## MDP value architecture

CB16 learns reusable value at several levels:

- `Qglobal[behavior_state][action]`
- `Qhc[hc_id][behavior_state][action]`
- `Qhd[hd_id][behavior_state][action]`
- `Qot[ot_id][behavior_state][action]`
- `Qtask[assignment_id][coarse_task_state][action]`

Only components with prior support contribute to the weighted decision score. The global behavior state intentionally excludes the task ID and taxonomy IDs so experience can be reused across tasks.

## Cloud use

1. Put the `chembreak16/` directory in the GitHub `ChemBreak` repository.
2. Open `notebooks/chembreak16_Cloud_Notebook.ipynb` in Google Cloud Notebook Enterprise.
3. Edit the user configuration cell if needed.
4. Run from Cell 1 downward.

Runtime state is isolated under `/content/chembreak16_storage`.

## Source task identifiers

The source bank preserves its original assignment and matrix identifiers (for example `CBV15C-...` and `V15C-...`). These are immutable task/provenance identifiers from the frozen source data, not references to a CB15 runtime package. Renaming them would alter task identity and invalidate the source/manifest hashes.
