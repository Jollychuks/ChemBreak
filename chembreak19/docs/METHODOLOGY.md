# ChemBreak19 methodology

## Experimental phases

The exact same 24 assignments are evaluated at Baseline, in three learning epochs, and in Optimized evaluation. Baseline is one direct target turn. Learning uses at most four target turns per task with base epsilon 0.30, 0.20, and 0.15. After Epoch 3 the learned artifacts are frozen. Optimized uses epsilon 0 and performs no learning.

Every phase/epoch is a separate fresh target conversation. Only learned Q-values and memory persist.

## Model roles

GPT-5.6 Sol is the Attack LLM, ChemDFM-v1.5-8B is the target, and Gemini 2.5 Flash is the single Judge LLM. A deterministic Python gate applies the fixed success thresholds to judge scores.

## Hierarchical Q-policy

The policy separates global behavior value, HC/HD/OT context value, and a task residual. Only supported components contribute. Each component's contribution is confidence-shrunk until it reaches the configured support target, reducing the influence of one-off observations. Exploitation can select only supported actions; otherwise the policy reports exploration or cold start.

## Candidate evidence

Every target-tested learning candidate is still logged with exact realization identity, action identity, state keys, reward, outcome, and post-update Q evidence. This layer is diagnostic in CB19 and is not used for late-turn replay.

## Successful trajectory memory

A trajectory is the ordered sequence of exact realized messages and their abstract actions from the beginning of a learning episode through success. The first successful observation creates the trajectory with one successful attempt.

At the beginning of a later learning episode for the same task, the highest-ranked successful trajectory is replayed from Step 1. Replay consumes normal target turns because ChemDFM is actually queried, but it does not call the Attack LLM. If success occurs during replay, the trajectory gains another successful observation. If all stored steps are replayed without success, the trajectory gains a failure observation and the remaining target-turn budget returns to the Q-policy.

If the later episode then succeeds using fresh fallback actions, the complete hybrid path actually used in that successful episode is stored as a new successful trajectory.

## Trajectory ranking

Reliability is the Wilson lower confidence bound of observed trajectory success frequency. Support grows with attempts and saturates at the configured target. Mean cumulative reward and mean Q evidence are bounded before combination. The default score is:

`0.55 × reliability + 0.20 × support + 0.15 × reward_quality + 0.10 × q_quality`

This prevents a 1/1 trajectory from automatically outranking a more repeatedly tested trajectory such as 2/3.

## Provider policy blocks

A provider safety rejection is recorded separately from ChemDFM behavior. It consumes no target turn and causes no Q, candidate-evidence, or trajectory-memory update. Another abstract action is tried when available. If all actions are blocked, the episode ends as non-success with terminal reason `attack_llm_policy_blocked_all_actions`.

## Freeze and Optimized

The freeze snapshot contains the hierarchical Q-policy, candidate evidence, trajectory memory, trajectory rankings, and experiment identity. Optimized begins from a fresh conversation. The highest-ranked frozen successful trajectory is replayed from the start. If it fails and turns remain, the frozen Q-policy chooses fallback actions and the Attack LLM realizes them. Q-values and all memory remain immutable.

## Metrics

Primary ASR uses all scheduled tasks as the denominator. CB19 also reports ASR@1, ASR@4, recovery rate, success retention, trajectory coverage, provider-block diagnostics, and a secondary target-evaluable ASR that excludes episodes in which no ChemDFM query occurred because every Attack-LLM action was provider-blocked. `baseline_audit.csv` preserves exact Baseline responses and judge scores for direct inspection.
