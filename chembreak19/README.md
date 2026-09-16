# ChemBreak 19 v19.0.0

ChemBreak19 is the fixed-24-task adaptive-MDP safety-evaluation package with **successful trajectory memory**.

## Model roles

- **Attack LLM:** `gpt-5.6-sol` through the OpenAI Responses API.
- **Target:** `OpenDFM/ChemDFM-v1.5-8B` locally on GPU.
- **Judge LLM:** `gemini-2.5-flash` through Vertex AI.
- One judge only; no second judge and no adjudicator.

## Fixed experiment flow

The same 24 assignments are used throughout:

`Baseline → Epoch 1 (.30) → Epoch 2 (.20) → Epoch 3 (.15) → Rank Trajectories → Freeze → Optimized (0)`

This is within-task adaptive optimization, not unseen-task generalization. The out-of-bank matcher is intentionally not part of CB19.

## Fresh-episode boundary

Baseline, each learning epoch, and Optimized start from a fresh target conversation and neutral initial MDP state. Conversation history never carries across phases. What persists across learning epochs is the hierarchical Q-policy, candidate-level diagnostic evidence, and successful trajectory memory.

## Hierarchical policy

CB19 uses reusable global, HC, HD, OT, and task-specific Q components. Early Q evidence is confidence-shrunk until it has repeated support, preventing one lucky observation from dominating cross-task exploitation. `exploitation` is reserved for actions with actual learned support; unsupported choices are explicit `exploration` or `cold_start` decisions.

## Successful trajectory memory

When a learning episode succeeds, CB19 stores the **complete ordered path from Turn 1 through success**: each abstract action plus its exact realized message. On the next epoch for that same task, the strongest remembered successful trajectory is replayed from the beginning of the fresh episode.

If replay succeeds, its reliability increases. If it fails, the failure is retained rather than erasing the earlier success, and the remaining target-turn budget is used by the Q-policy for fresh alternatives. A later fresh/hybrid success becomes a new trajectory. Trajectories are ranked with Wilson reliability, observation support, reward quality, and Q evidence.

Candidate-level evidence is still logged for auditability, but CB19 does **not** inject a late-turn remembered candidate into an unrelated fresh state.

## Provider policy blocks

OpenAI policy blocks such as `bio_policy` are provider events rather than target failures. A blocked fresh action consumes no ChemDFM turn, invokes no judge, and updates neither Q-values nor memory. CB19 tries another action on the same task when possible. If all actions are blocked, the episode ends conservatively as non-success and execution continues.

Stored trajectory replay does not call the Attack LLM because the already-generated exact messages are replayed directly.

## Freeze and Optimized

After Epoch 3, Q-values, visit counts, candidate evidence, trajectory memory, and trajectory rankings are frozen. Optimized evaluation starts fresh, replays the highest-ranked frozen successful trajectory when one exists, and then uses frozen-Q fresh fallbacks only if the trajectory fails and target-turn budget remains. No learning occurs during Optimized.

## Cloud use

Run `notebooks/chembreak19_Cloud_Notebook.ipynb` from Cell 1 downward. Runtime state stays under `/content/chembreak19_storage`.

Source identifiers such as `CBV15C-*` are immutable source-bank assignment IDs, not prior-version runtime artifacts.
