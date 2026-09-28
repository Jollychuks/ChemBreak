# ChemBreak27 Package Contents

ChemBreak27 v27.1.0 is a standalone task-local research package for ChemDFM and ChemLLM.

## Experimental architecture

- Every target-task pair receives a fresh Q-policy, visit table, stagnation table, route memory, and conversation state.
- Learning persists across three epochs of the same task only.
- The response state distinguishes hard refusal, soft refusal, safe redirection, limited engagement, partial assistance, substantive assistance, full harmful compliance, and off-objective behavior.
- The reward combines normalized CHCS movement, behavioral-class movement, a success bonus, a CHCS-1 penalty, a stagnation penalty, and a turn cost.
- Exact current-task route replay begins in Epoch 2 and uses the same five-turn budget as adaptive recovery.
- Policy and route memory freeze before terminal evaluation.
- Terminal replay cannot query a target after a failed intent-fidelity gate.

## Result integrity

- Scientific metrics use only queries linked to persisted turns in completed episodes.
- Issued but uncommitted queries remain immutable audit evidence and are counted separately.
- Restart recovery removes turn links from incomplete baseline, learning, and terminal episodes.
- Tests reproduce an interrupted CHCS 5 response and confirm that it cannot enter ASR, discovery, terminal, reproduction, CHCS-distribution, or query-budget results.

## Package identity

- Package version: 27.1.0
- Experiment revision: `CB27_INDEPENDENT_TASK_LOCAL_PROMPTS28_V1`
- Assignment namespace: `CB27P-*`
- Route namespace: `cb27-route`
- Provider seed: 27027

## Dependency integrity

- `google-genai==2.24.0`
- `google-auth==2.56.0`

The Google authentication pin satisfies the declared Google Gen AI dependency range. The cloud notebook checks the pin before installation and keeps pip output visible.

## Package hygiene

The release archive excludes Python bytecode, test caches, runtime databases, generated results, model caches, and external experiment artifacts.
