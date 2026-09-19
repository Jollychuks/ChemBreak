# ChemBreak20 methodology

## Experimental unit

The fixed panel contains 24 source tasks. Assignment IDs are immutable identifiers inherited from the source bank. Each target model is an independent experiment with its own controller state.

## Immutable task lock

For every task, ChemBreak20 stores the source-provided `original_prompt`, `original_goal`, `original_behavioral_intent`, and `requested_outcome`. These fields never change during the experiment. The Attack LLM may alter wording, framing, representation, sequencing, or the selected abstract strategy, but it may not change the requested behavioral objective.

Every generated candidate is checked by the Judge LLM before a target query. A candidate must meet the predeclared goal-preservation and task-fidelity thresholds. Drifted candidates are rejected and do not consume a target query.

## Hierarchical MDP

The state summarizes observed target behavior rather than hidden model state. The policy maintains global, HC-, HD-, OT-, and task-specific value components. Support-confidence shrinkage prevents a single observation from receiving full cross-task influence.

The action set is fixed before the run. The Attack LLM realizes the selected abstract action; it does not choose the action.

## Abstract route memory

A route is an ordered sequence of abstract actions observed in one task episode. Route memory records attempts, successes, failures, reward, progress, support, and epoch observations. Exact generated candidate text is logged for audit but is not replayed as learned memory.

In later learning epochs, routes provide a bounded bonus to actions that continue a previously observed route prefix. The normal epsilon-greedy policy remains active, so route evidence guides rather than dictates learning.

## Learning protocol

Each of the three learning epochs contains one fresh conversational episode per task, with at most four target turns. Target conversational memory resets between epochs. Controller memory persists.

Learning orders are deterministically shuffled from the fixed seed and recorded in the checkpoint metadata. This controls a systematic task-position effect while remaining reproducible.

## Success

The Judge LLM returns measurements only. Python computes final success from fixed thresholds covering candidate goal preservation, goal progress, response task fidelity, response correctness, chemistry relevance, and the declared safety-boundary measurements. Thresholds are fixed before live execution.

## Terminal exploitation

After all three learning epochs, Q-values and route memory are frozen. The final evaluation starts from a fresh target context for each task and allows at most four target turns:

1. candidate informed by the highest-ranked learned route;
2. candidate informed by the next distinct learned route;
3. frozen-policy synthesis using accumulated route evidence;
4. a distinct frozen-policy synthesis.

The Attack LLM may realize new text, but Q-values, route statistics, and rankings do not change. Stop immediately on verified success.

## Targets

ChemDFM and ChemLLM are run sequentially so only one large local target is resident on the GPU. No Q-value or route evidence transfers between targets in the main experiment. Each target uses its own documented/native dialogue template: ChemDFM uses its round-based Human/Assistant format, while ChemLLM uses the tokenizer-native InternLM2 chat template. This avoids a target-format confound.

## Query-budget interpretation

Baseline has one target query. The adaptive procedure has at most 16 target queries across the three learning epochs plus terminal exploitation. ChemBreak20 therefore reports both frozen terminal ASR@1..4 and a cumulative discovery curve at 1/4/8/12/16 queries. An equal-budget non-adaptive control is implemented separately for publication ablation.
