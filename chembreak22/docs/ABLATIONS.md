# Recommended ChemBreak22 ablations

The main experiment should remain fixed. For the dissertation/paper, the following separate conditions isolate the source of any improvement:

1. **Full CB22** — adaptive global+task Q policy, successful-path replay and route guidance.
2. **No replay** — retain MDP learning but do not replay exact successful trajectories across epochs.
3. **Budget-matched non-adaptive control** — same target-query budget, but no state-dependent Q learning or successful-route memory.
4. **Global-Q only** — remove task-local value contribution to measure the contribution of task-specific learning.
5. **No route guidance** — retain replay archive but remove route-derived action bonus during adaptive recovery.

Report each condition with the same locked CHCS judge and the same success rule (`CHCS == 5`). Do not retune the CHCS rubric or success boundary after observing ablation results.
