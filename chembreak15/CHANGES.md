# ChemBreak 15 — 15.0.0

ChemBreak 15 is a methodological revision focused on preventing the action lock-in observed during the preceding development run while keeping the exact same 24-task panel for comparability.

## Policy changes

- Replaces task-wide `Q_task[task][action]` memory with state-aware `Q_task[task][state][action]` memory.
- Changes the base epsilon schedule from a steep decay to `0.30 → 0.20 → 0.15`.
- Adds bounded adaptive epsilon for novel states and immediate non-positive feedback, capped at `0.35`.
- Adds a non-positive repetition penalty to the action-selection score.
- Temporarily blocks an action after two consecutive non-positive repeats when another action is available.
- Uses policy schema version 2 and refuses to load older flat task-memory artifacts.

## Diagnostics

- Live adaptive-turn output now shows selection mode, base→effective epsilon, `Qg`, task-state `Qt`, combined Q, repetition penalty, adjusted score, and blocked actions.
- Adds `release/policy_diagnostics.csv` for turn-level audit of all policy decisions.
- Experiment identity now includes all CB15 exploration and repetition-control parameters so incompatible checkpoints cannot be silently mixed.

## Runtime isolation

- Uses `/content/chembreak15_storage` only.
- Uses `CB15_MDP_MINI24_V1` as the default experiment revision.
- Uses policy seed `15026`.
- Rejects mismatched or unverified CB15 checkpoints/policies.
- Does not migrate prior-version runtime state.

## Fixed mini dataset

- Retains the exact validated 24-task panel so the policy revision can be compared on identical tasks.
- Locks exact assignment IDs, assignment-ID hash, source-bank SHA-256, manifest SHA-256, reserve exclusion, and HC/HD/OT coverage.
- Keeps 24 Baseline episodes, 72 Learning episodes, and 24 Optimized Evaluation episodes.

## Structured-output reliability

- Retains the hardened Vertex structured-output adapter: SDK parsed objects are preferred, JSON-text fallback is validated, malformed/truncated outputs are retried with a larger output budget, and malformed judge values are never fabricated.
- Retains the extended live preflight probe so structured-output failures can be caught before the main run.
