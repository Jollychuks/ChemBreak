# ChemBreak17 methodology

## Experimental phases

The exact same 24 assignments are evaluated at Baseline, in three learning epochs, and in Optimized evaluation. Baseline is one direct target turn. Learning uses at most four adaptive turns per task with base epsilon 0.30, 0.20, and 0.15. After Epoch 3 the learned artifacts are frozen. Optimized uses epsilon 0 and performs no learning.

## Model roles

GPT-5.6 Sol is the attack LLM. It realizes an abstract MDP action as one non-operational benchmark message. ChemDFM-v1.5-8B is the target. Gemini 2.5 Flash is the single judge LLM. A deterministic Python gate applies the fixed success thresholds to judge scores.

## Hierarchical Q-policy

The Q-policy separates reusable global behavior value, HC/HD/OT context value, and a small task residual. Only components with prior support contribute to the weighted action score.

## Evidence memory

CB17 distinguishes the abstract action from the exact realized candidate. Each normalized attack-LLM utterance has a SHA-256-prefix `realization_id`. An action-conditioned `candidate_id` hashes the abstract action together with that realization. This preserves exact-text identity while preventing the same utterance produced under two different MDP actions from being silently conflated. For every learning attempt, the memory records task, candidate, action, state keys, epoch, turn, outcome, reward, and Q evidence.

A remembered candidate is never treated as permanently successful. A later failure updates the same candidate evidence instead of deleting its earlier success.

## Ranking rule

Candidate reliability is the Wilson lower confidence bound of observed success frequency. Support increases with the number of observations and saturates at the configured target support. Mean reward and mean Q are transformed to bounded quality values. The default frozen score is:

`0.55 × reliability + 0.20 × support + 0.15 × reward_quality + 0.10 × q_quality`

This prevents a one-observation 100% rate from automatically dominating a more repeatedly tested candidate.

## Learning use of memory

Epsilon exploration always creates a fresh attack-LLM realization. During exploitation, CB17 may replay the strongest untried previously successful exact candidate for that task. The resulting success or failure updates both the Q-policy and evidence memory.

## Freeze and Optimized

The freeze snapshot contains the full hierarchical Q-policy, visit counts, evidence memory, candidate rankings, and experiment identity. Optimized reads this snapshot without modifying it.

For each optimized turn, CB17 first tries the highest-ranked untried frozen candidate, preferring candidates previously seen in the current behavioral state. If no frozen candidate remains, the frozen Q-policy selects an action and GPT-5.6 Sol realizes a fresh fallback. A failed optimized attempt can change the observed state but cannot change any learned parameter.

## Metrics

CB17 reports phase ASR, ASR@1, ASR@4, mean turns, mean reward, recovery rate after a failed first turn, success retention across phases, evidence coverage at freeze, and hierarchical-policy support diagnostics.
