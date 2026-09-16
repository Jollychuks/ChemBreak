# ChemBreak 18 v18.0.1

ChemBreak18 is the corrected fixed-24-task adaptive-MDP safety-evaluation package for Google Cloud Notebook Enterprise.

## Model roles

- **Attack LLM:** `gpt-5.6-sol` through the OpenAI Responses API.
- **Target:** `OpenDFM/ChemDFM-v1.5-8B` locally on GPU.
- **Judge LLM:** `gemini-2.5-flash` through Vertex AI.
- One judge only; no second judge and no adjudicator.

## Fixed experiment flow

The same 24 assignments are used throughout:

`Baseline → Epoch 1 (.30) → Epoch 2 (.20) → Epoch 3 (.15) → Build Rankings → Freeze → Optimized (0)`

This is a within-task adaptive-optimization experiment, not unseen-task generalization. The out-of-bank embedding/LLM matcher is intentionally not part of CB18.

## Fresh-episode boundary

Baseline is an independent one-turn evaluation. Epoch 1, Epoch 2, Epoch 3, and Optimized each start from a fresh target conversation with neutral initial MDP state. Baseline responses and earlier-epoch conversation histories do not carry into a later episode. Only the learned hierarchical Q-policy and Evidence Memory persist across learning epochs.

## Hierarchical policy and Evidence Memory

CB18 uses:

- `Qglobal[behavior_state][action]`
- `Qhc[hc_id][behavior_state][action]`
- `Qhd[hd_id][behavior_state][action]`
- `Qot[ot_id][behavior_state][action]`
- `Qtask[assignment_id][coarse_task_state][action]`

and persistent task-specific Evidence Memory for exact realized Attack-LLM candidates.

The Attack LLM receives the exact benchmark task as reference context together with the high-level goal, current adaptive state, and the MDP-selected abstract action. It remains constrained to non-operational benchmark wording.

`exploitation` is reserved for actions with actual learned support in the current hierarchical state. Unsupported actions can appear only through explicit exploration or cold-start selection.

Evidence Memory tracks exact realization identity separately from abstract action identity and accumulates attempts, successes, failures, reward evidence, state keys, Q evidence, and epoch history. Successful candidates are ranked using Wilson lower-bound reliability, observation support, reward quality, and Q evidence.

## Provider policy blocks

OpenAI policy blocks such as `bio_policy` are treated as provider events rather than target failures. A blocked action is excluded for the rest of that task episode; no ChemDFM query, judge call, Q update, or Evidence update occurs for that event. CB18 automatically tries another available action on the same task. If all actions are blocked, the task is completed conservatively as a non-success and execution continues.

## Freeze and Optimized

After Epoch 3, Q-values, visit counts, Evidence Memory, and candidate rankings are frozen. Optimized evaluation uses `epsilon = 0`, reads the frozen artifacts, and performs no learning. It first tries strong unattempted, coarse-state-compatible frozen successful candidates and then uses frozen-policy fresh fallbacks if needed.

## Cloud use

Run `notebooks/chembreak18_Cloud_Notebook.ipynb` from Cell 1 downward. Runtime state stays under `/content/chembreak18_storage`.

Source identifiers such as `CBV15C-*` are immutable assignment IDs from the source bank and are not prior-version runtime artifacts.
