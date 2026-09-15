# ChemBreak 14 — 14.0.0

ChemBreak 14 is a clean namespace/version transition of the audited mini-dataset MDP experiment. Runtime code, paths, configuration, notebooks, package imports, checkpoints, policies, and documentation now use CB14/`chembreak14` consistently.

## Runtime isolation

- Uses `/content/chembreak14_storage` only.
- Uses `CB14_MDP_MINI24_V1` as the default experiment revision.
- Uses policy seed `14026`.
- Rejects mismatched or unverified CB14 checkpoints/policies instead of reusing them.
- Does not migrate or consume previous-version runtime state.

## Fixed mini dataset

- Retains the exact already-validated 24-task panel so the version change does not alter task membership.
- Locks exact assignment IDs, assignment-ID hash, source-bank SHA-256, manifest SHA-256, reserve exclusion, and HC/HD/OT coverage.
- Keeps 24 Baseline episodes, 72 Learning episodes, and 24 Optimized Evaluation episodes.

## Live notebook reporting

- Adds `LIVE_PROGRESS = True` to the user configuration cell.
- Baseline prints a live result line after its target/judge turn and a running Baseline ASR line after each completed task.
- Learning and Optimized Evaluation print one live line after every target/judge turn, plus a running phase-ASR line after each episode.
- Live output includes assignment ID, turn, abstract action ID, response class, success, goal progress, reward, running ASR, overall progress, query count, elapsed time, and ETA.
- Output is flushed immediately for Notebook Enterprise; prompt/response bodies are not printed.

## Structured-output reliability

- Keeps the hardened Vertex structured-output adapter: SDK parsed objects are preferred, JSON-text fallback is validated, malformed/truncated outputs are retried with a larger output budget, and malformed judge values are never fabricated.
- Keeps the extended live preflight probe so structured-output failures can be caught before the main run.
