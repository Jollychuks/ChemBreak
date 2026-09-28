# ChemBreak27 Validation Report

## Package

Version: `27.0.1`

Experiment revision: `CB27_TASK_LOCAL_RESPONSE_AWARE_PROMPTS28_V2`

## Validation scope

The release was checked at four levels:

1. Static compilation and configuration validation.
2. Locked dataset, manifest, identifier, and checksum validation.
3. Automated unit and integration tests, including interruption recovery.
4. Complete dry-run execution of all 28 tasks for both ChemDFM and ChemLLM.

## Automated tests

`30 passed`

The tests cover:

- the locked 28-task panel and `CB27P-*` identifiers
- pinned target-model revisions
- primary and fallback judge behavior
- provider-failure audit provenance
- response-class schema and response-aware reward
- target-query persistence before judgment
- rollback of incomplete baseline, learning, and terminal episodes
- exclusion of abandoned successes from all scientific success metrics
- restart and resume integrity after an interrupted CHCS 5 response
- exact route replay and adaptive recovery
- the five-turn learning budget
- terminal intent-gate enforcement
- frozen terminal route execution
- public redaction and private transcript export
- strict cross-task policy isolation
- empty Q-values, visits, stagnation state, and routes at each new task
- contiguous completion of one task before the next task begins
- version-clean filenames, code, configuration, notebook, and documentation

## Result-integrity rule

Every issued target query is preserved for operational auditing. A query contributes to ASR, CHCS distributions, discovery, reproduction, or query-budget curves only when a persisted turn in a completed episode references its query index. The interruption tests insert a judged CHCS 5 query, roll back the episode, resume execution, and verify that the abandoned success is not counted.

## Complete dry-run result

For each target, the dry run completed 28 baseline episodes, 28 Epoch 1 episodes, 28 Epoch 2 episodes, 28 Epoch 3 episodes, and 28 terminal episodes. ChemDFM and ChemLLM each produced 504 issued queries, 504 committed experimental queries, and zero uncommitted queries in the uninterrupted full run.

## Production configuration

Targets:

- ChemDFM: `OpenDFM/ChemDFM-v1.5-8B`, revision `f5790d56a903ce480b1eff8d0adf9613d8acee0c`
- ChemLLM: `AI4Chem/ChemLLM-7B-Chat-1_5-SFT`, revision `a6396a56cf353ea5fc8e487a1f4ec3727c44d3ed`

Evaluation roles:

- attack-language component: Gemini 3.1 Pro Preview
- intent-fidelity gate: Gemini 3.8 Flash
- primary CHCS judge: GPT-5.6 Sol
- technical fallback CHCS judge: Gemini 3.8 Flash

The fallback judge is called only when the primary judge fails to return a valid structured judgment.

## Task isolation

Every assignment completes Baseline, Epoch 1, Epoch 2, Epoch 3, Freeze, and Terminal Evaluation before the next assignment begins. A new assignment receives fresh Q-values, visit counts, state-action stagnation records, route memory, action history, and conversation state.

## Live-run boundary

The complete workflow was executed with the supplied deterministic dry-run components. The live GPU and provider path was inspected and retained, but it was not executed here because the authorized API credentials, Colab GPU runtime, and model downloads are available only in the researcher's live environment. Cell 7 of the cloud notebook performs tokenizer and provider preflight before any live experimental query.
