# ChemBreak 16

ChemBreak 16 is the next development iteration of the adaptive MDP safety-evaluation harness. It addresses two problems exposed by the CB15 run: overly sparse state reuse and the lack of an unseen-task generalization measurement.

## What changed

CB16 uses a hierarchical policy rather than two near-duplicate Q tables. The frozen policy contains a global behavior table, separate HC/HD/OT context tables, and a lightweight task-specific table. These components use different keys and are combined by a weighted mean over components with prior visits. A new `cold_start` decision mode explicitly marks states with no learned support.

CB16 also restores the fixed partition firewall. The 24 learning tasks come only from the previously locked `CB12_PARTITION_V1` **Train** split. A separate 12-task holdout comes only from **Test1**. Holdout tasks cannot be accessed by the runner until the policy has been frozen.

## Cloud workflow

1. Put the `chembreak16/` folder in the existing GitHub `ChemBreak` repository.
2. Open `chembreak16_Cloud_Notebook.ipynb` in Google Cloud Notebook Enterprise.
3. Confirm the first-cell `PROJECT_ID`, `REPO_URL`, `BRANCH`, `PROJECT_SUBDIR`, `EXPERIMENT_REVISION`, `LIVE`, and `LIVE_PROGRESS` values.
4. Run the notebook from Cell 1 downward.
5. Do not reuse CB15 storage or policy artifacts. CB16 writes under `/content/chembreak16_storage`.

## Experimental flow

The full run is Train Baseline (24) → three Train24 learning epochs (72) → freeze → Train24 optimized diagnostic (24) → Test1 Holdout Baseline (12) → Test1 Holdout Optimized (12). The final headline generalization comparison is Holdout Baseline ASR versus Holdout Optimized ASR.

The notebook displays live per-turn decisions and includes `Qg`, `Qhc`, `Qhd`, `Qot`, `Qt`, combined Q, active components, support visits, epsilon, repetition penalty, and blocked actions.

## Safety/evaluation scope

The policy layer operates on abstract action identifiers and safety-evaluation state. Chemistry-specific bypass transformations are intentionally not hard-coded into the policy implementation.
