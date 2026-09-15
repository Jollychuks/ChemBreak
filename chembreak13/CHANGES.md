# ChemBreak 13 changes from ChemBreak 12

- Uses a fixed taxonomy-balanced 24-task mini dataset rather than the 241-task Train split.
- Uses one target initially: ChemDFM.
- Baseline is a 24-task single-turn original-prompt run.
- Learning uses three fresh 24-task epochs with at most four target turns each.
- The MDP selects one strategy per turn; the actor no longer generates a three-candidate pool.
- Q-values start neutral; response-class-specific heuristic action bonuses are removed.
- Keeps both general-policy memory and task-specific memory across epochs.
- Final optimized phase starts from the learned policy rather than automatically resending the original benchmark prompt.
- Policy is frozen before optimized evaluation; no Q updates occur in the final phase.
- Total headline episode count is 120; maximum target queries are 408.
- Storage and caches are CB13-only under `/content/chembreak13_storage`.
