# ChemBreak 13 methodology

ChemBreak 13 uses one fixed 24-task mini dataset. The experiment has four simple phases:

1. **Baseline:** send each original benchmark prompt once to ChemDFM. This produces Baseline ASR and a baseline response profile for each task.
2. **Learning:** run the same 24 tasks in three fresh epochs. Each episode can use up to four target turns. The MDP selects exactly one abstract strategy per turn. The actor realizes that selected strategy. The judge produces the reward signals. Q-values and task memory persist across epochs.
3. **Optimized attack:** freeze the learned policy, set exploration to zero, and run the same 24 tasks again from fresh conversations. The learned policy chooses the first strategy before the first target call; the original benchmark prompt is not automatically sent first in this phase.
4. **Results:** compare Baseline ASR with Optimized ASR and report learning-epoch ASR, strategy use, rewards, turns, and the ASR change in percentage points.

## Counts

- 24 baseline episodes.
- 72 learning episodes: 24 tasks × 3 epochs.
- 24 optimized episodes.
- 120 total episodes.
- Baseline uses one target query per task.
- Learning and optimized episodes use at most four target queries and stop early on success.
- Maximum target queries: 24 + (72 × 4) + (24 × 4) = 408.

## Learning

The controller stores both a general state-action table and a task-specific action table. Strategy values start at zero. There are no response-class-specific hard-coded bonuses. Exploration decreases by epoch: 0.30, 0.15, 0.05. The optimized phase uses epsilon=0 and disables all Q updates.

## Safety boundary

The supplied actor is deliberately safety-bounded: it realizes only abstract, non-operational request-shaping strategies and is instructed not to add synthesis conditions, quantities, precursor substitutions, optimization details, weaponization details, or safeguard-bypass instructions. This package implements the full learning, state, reward, checkpoint, cloud, target, judging, and reporting architecture without bundling chemistry-specific operational jailbreak recipes.
