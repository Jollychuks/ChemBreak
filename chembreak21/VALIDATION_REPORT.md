# ChemBreak21 v21.0.0 — Validation Report

Validation date: 2026-09-19

## Package invariants

- Namespace: `CB21`
- Experiment revision: `CB21_REPLAY_MDP_PROMPTS28_V1`
- Dataset: 28 unique rows from `data/prompts.csv`
- Attack LLM: `gemini-3.1-pro-preview`
- Judge LLM: `gemini-3.8-flash`
- Targets: ChemDFM and ChemLLM, with separate controller state
- Learning epochs: 3
- Learning turn budget: 4 target turns per task per epoch
- Epsilon schedule: 0.30, 0.20, 0.15
- Exactly one notebook: `notebooks/chembreak21_Cloud_Notebook.ipynb`
- Notebook saved outputs: 0
- Notebook saved execution counts: 0

## Regression tests

`pytest -q`: **29 passed**.

The regression suite specifically checks:

- locked 28-prompt source and manifest hashes;
- no re-enabling of provider-blocked actions;
- the last provider-valid action remains available even when repetition suppression would otherwise block it;
- exact prompt-path storage and cross-epoch reliability updates;
- confirmed-route status after success in distinct epochs;
- two-turn replay failure leaves two adaptive recovery turns;
- four-turn replay failure leaves no recovery turns;
- Final executes the complete first route, resets, then executes the complete second route;
- raw target responses are persisted before judge completion;
- raw target-query audit rows survive episode rollback;
- notebook local-package import activation is present;
- no OpenAI API key/model dependency remains in the notebook.

## Full dry-run integration

A complete two-target mock run was executed with all 28 tasks.

Per target:

- episodes: 140 = 28 baseline + 84 learning + 28 final;
- learning Q updates: 336;
- route observations: 84;
- actual target queries in failure-only mock: 420;
- saved judged turns: 420.

The same completed checkpoints were run again and remained at exactly the same counts, confirming that normal resume does not duplicate completed learning or target calls.

## Important live-run boundary

This validation does not execute the real Gemini APIs or load the full ChemDFM/ChemLLM weights. The cloud notebook preflight performs environment-specific Vertex model probes, target tokenizer probes, CUDA/BF16 checks, and then the real live run.
