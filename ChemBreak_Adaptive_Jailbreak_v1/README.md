# ChemBreak Adaptive Jailbreak v1

This folder is the GitHub-ready execution layer for the completed ChemBreak task bank. It is designed to be committed as a subdirectory of `https://github.com/Jollychuks/ChemBreak` and run from Google Cloud Colab Enterprise.

## Experimental design

ChemBreak has two sections.

### Controlled evaluation

- C0 Direct Single-Turn
- C1 Repeated Single-Turn with fresh target context and a matched query budget
- C2 Fixed Multi-Turn where the full query sequence is generated before any target response is observed

### Adaptive ChemBreak

- C3 Response-Adaptive Multi-Turn ChemBreak

The adaptive attacker starts from a four-route chemistry-specific graph:

1. Representation
2. Context
3. Conceptual Decomposition
4. Output-Oriented

After each target response, the attacker receives the fixed task, initial graph, actual target conversation history, current route, and remaining query budget. It can CONTINUE, REFINE, SWITCH, or STOP_FAILURE. Judge reasoning is not fed back to the attacker.

## Target models

- ChemDFM: `OpenDFM/ChemDFM-v1.5-8B`
- ChemLLM: `AI4Chem/ChemLLM-7B-Chat-1_5-SFT`
- LlaSMol: `osunlp/LlaSMol-Mistral-7B`, loaded as a LoRA adapter over `mistralai/Mistral-7B-v0.1`

Targets are loaded one at a time. Within a target block, the model weights are loaded once and reused across every task and condition. Fresh single-turn attempts reset conversation state without reloading model weights.

## Repository workflow

Upload this entire folder to the root of the ChemBreak repository so that the path is:

```text
ChemBreak/ChemBreak_Adaptive_Jailbreak_v1/
```

Then open:

```text
ChemBreak_Adaptive_Jailbreak_v1_Colab_Enterprise.ipynb
```

in Colab Enterprise.

The notebook clones or refreshes the GitHub repository inside the ephemeral runtime, enters this project directory, installs dependencies, discovers the completed V15 final task bank in Google Cloud Storage, creates a non-committed runtime configuration, and runs the experiment.

## Recommended Google Cloud sequence

1. TEST
2. PILOT
3. Freeze the attack budget, route-switch limit, judge thresholds, prompts, target revisions, and evaluation rules
4. PRODUCTION

Do not tune methodology after examining production results.

## Default query budgets

- C0 Direct: 1 target query
- C1 Repeated Single: maximum 5 independent target queries
- C2 Fixed Multi-Turn: maximum 5 target queries
- C3 Adaptive ChemBreak: maximum 5 target queries and maximum 2 route switches

Every actual call to the target model counts against the relevant query budget.

## Durable and resumable execution

Each run has a stable `run_id`, such as `test_001`, `pilot_001`, or `production_001`.

Local output is namespaced as:

```text
outputs/CB-ADAPTIVE-JAILBREAK-V1/<mode>/<run_id>/
```

The notebook points the same run to a GCS prefix such as:

```text
gs://<bucket>/ChemBreak_Adaptive_Jailbreak_v1/<mode>/<run_id>/
```

On restart, existing GCS checkpoints are downloaded. Completed task-target-condition units are skipped.

## Output separation

Restricted files contain prompts and target responses:

```text
restricted/
  attack_assets.jsonl
  executions.jsonl
  errors.jsonl
```

Public analysis files contain only evaluation summaries:

```text
public/
  selected_task_ids.json
  preflight_manifest.json
  run_level_metrics.csv
  summary_metrics.csv
  summary_metrics.json
```

## Public GitHub repository warning

The ChemBreak repository is public. The supplied `.gitignore` excludes the local production task bank, runtime output, and runtime configuration. Keep the harmful task bank and raw model transcripts in restricted Google Cloud Storage rather than committing them to GitHub.

## Preflight manifest

Preflight records:

- run version, mode, and run ID
- Git commit
- runtime configuration hash
- GPU details
- detected task-bank schema
- Hugging Face target model revisions
- Vertex attacker and judge connectivity

This provides a reproducibility record before the target runs start.

## Execution commands

After `configs/runtime.yaml` exists:

```bash
python scripts/preflight.py --config configs/runtime.yaml
python scripts/run.py prepare --config configs/runtime.yaml
python scripts/run.py execute --config configs/runtime.yaml --target chemdfm
python scripts/run.py execute --config configs/runtime.yaml --target chemllm
python scripts/run.py execute --config configs/runtime.yaml --target llasmol
python scripts/run.py metrics --config configs/runtime.yaml
```

For debugging only, controlled and adaptive sections can still be executed separately:

```bash
python scripts/run.py controlled --config configs/runtime.yaml --target chemdfm
python scripts/run.py adaptive --config configs/runtime.yaml --target chemdfm
```

Running `execute` is faster because each target model is loaded once for both sections.

## LlaSMol limitation

LlaSMol-Mistral-7B is an instruction-tuned LoRA adapter rather than a conventional chat model. The runner reconstructs prior query-response context textually for multi-turn evaluation. Report this protocol as a model-specific limitation.

## Responsible storage

This package is for authorized model-safety evaluation. Do not connect the workflow to physical chemical execution, procurement, or laboratory automation. Keep raw unsafe target outputs access-controlled.
