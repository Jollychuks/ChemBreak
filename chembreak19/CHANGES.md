# ChemBreak 19 — v19.0.0

## Main methodological change

- Replaces late-turn exact-candidate replay with **successful trajectory memory**.
- A successful path is stored as the full ordered sequence of actions and exact realized messages from Turn 1 through success.
- Later learning epochs replay the best remembered trajectory from the beginning of a fresh episode.
- A replay failure is retained as negative evidence; remaining turns return to the hierarchical Q-policy and fresh Attack-LLM generation.
- Fresh or hybrid successes are stored as new trajectories.
- Optimized uses frozen trajectory rankings first, then frozen-Q fresh fallback without learning.

## Additional hardening

- Keeps the same fixed 24 tasks, ChemDFM target, GPT-5.6 Sol Attack LLM, Gemini 2.5 Flash single Judge LLM, success thresholds, reward configuration, and epsilon schedule.
- Keeps Baseline, every learning epoch, and Optimized as separate fresh conversations.
- Adds support-confidence shrinkage to hierarchical Q combination so a one-off success cannot dominate cross-task exploitation.
- Keeps candidate-level evidence for diagnostics but disables candidate replay.
- Keeps OpenAI provider-policy blocks non-retryable and non-terminal when alternate actions remain.
- Corrects the final provider-block log to show `continuing=NO` when all actions are exhausted.
- Adds target-evaluable ASR as a secondary metric while preserving all-scheduled-task ASR as the primary metric.
- Uses experiment revision `CB19_TRAJECTORY_MDP_MINI24_V1` and fresh CB19 storage.
