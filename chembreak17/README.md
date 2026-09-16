# ChemBreak 17

ChemBreak17 is a fixed-24-task adaptive-MDP safety-evaluation package for Google Cloud Notebook Enterprise.

## Model roles

- **Attack LLM:** `gpt-5.6-sol` through the OpenAI Responses API.
- **Target:** `OpenDFM/ChemDFM-v1.5-8B` locally on GPU.
- **Judge LLM:** `gemini-2.5-flash` through Vertex AI.
- No second judge and no adjudicator.

## Fixed experiment flow

The same 24 assignments are used throughout:

`Baseline → Epoch 1 (.30) → Epoch 2 (.20) → Epoch 3 (.15) → Build Rankings → Freeze → Optimized (0)`

This is a within-task adaptive-optimization experiment, not unseen-task generalization. The out-of-bank embedding/LLM matcher is intentionally not part of CB17.

## CB17 learning architecture

CB17 keeps the hierarchical value system:

- `Qglobal[behavior_state][action]`
- `Qhc[hc_id][behavior_state][action]`
- `Qhd[hd_id][behavior_state][action]`
- `Qot[ot_id][behavior_state][action]`
- `Qtask[assignment_id][coarse_task_state][action]`

and adds persistent **Evidence Memory** for exact realized attack-LLM candidates. The memory stores an exact-text `realization_id` and a separate action-conditioned `candidate_id`, so the generated text and the abstract action are never conflated and tracks attempts, successes, failures, rewards, state keys, Q evidence, and epochs observed.

During learning, remembered successful exact candidates may be replayed during exploitation. Later failures reduce their evidence but do not erase earlier successes. Exploration still requests fresh realizations from GPT-5.6 Sol.

After Epoch 3, successful candidates are ranked using a fixed weighted rule based on Wilson lower-bound reliability, support, reward quality, and Q evidence. The Q-policy, evidence memory, and candidate rankings are then frozen.

Optimized evaluation first tries unattempted frozen successful candidates, preferring state-compatible entries. If they are exhausted, the frozen Q-policy selects an action and the attack LLM produces a fresh fallback. Optimized evaluation never updates the Q-policy, evidence memory, or rankings.

## Cloud use

Run `notebooks/chembreak17_Cloud_Notebook.ipynb` from Cell 1 downward. Runtime state stays under `/content/chembreak17_storage`.

Source identifiers such as `CBV15C-*` are immutable assignment IDs from the source bank and are not CB15 runtime artifacts.
