# ChemBreak27 v27.0.1 Release Notes

This release is a complete task-local production package for ChemDFM and ChemLLM.

## Scientific integrity corrections

- Scientific metrics use only target queries linked to committed turn records.
- Issued but uncommitted queries remain available as immutable audit evidence and are explicitly counted.
- Restart recovery removes all turn links from incomplete baseline, learning, and terminal episodes.
- Automated tests reproduce an interrupted CHCS 5 query and confirm that it cannot enter ASR, discovery, terminal, reproduction, CHCS-distribution, or query-budget results.

## Methodology implementation

- Every target-task pair receives a fresh Q-policy, visit table, stagnation table, route memory, and conversation state.
- Learning persists across three epochs of the same task only.
- The response state distinguishes hard refusal, soft refusal, safe redirection, limited engagement, partial assistance, substantive assistance, full harmful compliance, and off-objective behavior.
- The reward combines normalized CHCS movement, behavioral-class movement, a success bonus, a CHCS-1 penalty, a stagnation penalty, and a turn cost.
- Exact current-task route replay begins in Epoch 2 and uses the same five-turn budget as adaptive recovery.
- Policy and route memory freeze before terminal evaluation.
- Terminal replay cannot query a target after a failed intent-fidelity gate.

## Package identity and hygiene

- Package version is 27.0.1.
- Assignment IDs use the `CB27P-*` namespace.
- Route identifiers and provider seeds use ChemBreak27 values.
- Generated caches and stale test artifacts are excluded from the archive.
