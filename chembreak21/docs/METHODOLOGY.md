# ChemBreak21 methodology

## Research interpretation

ChemBreak21 measures **within-task adaptive discovery, cross-epoch replication, recovery, and final exploitation**. Because the same 28 source prompts are used during learning and final evaluation, the experiment should not be described as unseen-prompt generalization.

## State and action model

The MDP uses a compact behavioral state derived from target response class, goal-progress bin, fidelity bin, response-correctness bin, recent quality trend, reward sign and turn stage. The learned policy has two value layers:

- global state/action value;
- task-local state/action value.

The previous HC/HD/OT layers are deliberately removed because the uploaded prompt dataset contains only one functional and one semantic category.

## Successful route memory

A route stores the exact sequence of attacker prompts together with the abstract MDP actions that produced them. Each route accumulates attempts, successes, failures, success/failure epochs, reward, quality progress and turns-to-success. A route is:

- **discovered** after at least one successful episode;
- **confirmed** after success in at least two distinct epochs.

Confirmed routes are ranked above one-off discoveries.

## Replay and recovery

If a two-turn route discovered in Epoch 1 is replayed in Epoch 2 and fails at turn 2, the two unused turns become adaptive recovery turns. If a four-turn route fails on replay, no recovery turns remain in that epoch. Failed replay reduces route reliability but does not erase the route.

## Final stage

After freeze, no Q-value or route statistic is updated. Each ranked successful route is executed in a fresh target conversation and the **entire stored path** is replayed. After the configured number of stored routes are exhausted, two new terminal candidates may be generated from frozen evidence. These do not become learning updates.

## Query accounting

Every target response is written to the immutable `target_queries` audit table before the Judge LLM is called. Therefore target-query counts include calls whose subsequent judge evaluation fails. Technical/provider errors are reported separately from target success/failure.
