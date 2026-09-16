# ChemBreak 18 — v18.0.1

## Audit hardening over v18.0.0

- Keeps the exact 24-task panel, ChemDFM target, GPT-5.6 Sol Attack LLM, Gemini 2.5 Flash single Judge LLM, success thresholds, reward settings, hierarchical Q architecture, and epsilon schedule unchanged.
- Keeps Baseline, each learning epoch, and Optimized as separate fresh conversations.
- Requires exact-candidate replay to match the current coarse task state so context-dependent successes are not replayed out of context.
- Makes Evidence Memory replay respect the policy repetition block.
- Records post-update learned Q evidence for candidate ranking while retaining pre-update selection Q in diagnostics.
- Handles OpenAI policy errors returned in a failed Responses API object as non-retryable provider events, not just policy errors raised as HTTP exceptions.
- Uses experiment revision `CB18_EVIDENCE_MDP_MINI24_V2` so it cannot resume a v18.0.0/V1 checkpoint.
