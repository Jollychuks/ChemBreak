# ChemBreak Colab Generator V4.2

ChemBreak V4 is a controlled four-family generator and judge comparison framework for selecting the models that should later be used to build and evaluate the final ChemBreak task bank.

## Crossed four-family design

Every family acts as both a generator and a blind judge:

- A generates, then A, B, C, and D judge.
- B generates, then A, B, C, and D judge.
- C generates, then A, B, C, and D judge.
- D generates, then A, B, C, and D judge.

This creates all 16 generator-judge pairings.

## Current model registry

- Family A: Qwen, `Qwen/Qwen3.5-9B`
- Family B: Mistral, `mistralai/Ministral-3-8B-Instruct-2512-BF16`
- Family C: Gemma, `google/gemma-4-E4B-it`
- Family D: Phi, `microsoft/phi-4`

The exact representatives are centralized in `model_registry.json`.

## Shared experimental controls

All generators receive the same:
- generation matrix
- persisted scenario plan
- taxonomy definitions
- generator prompt
- candidate count
- decoding policy
- structural validation rules
- corresponding numeric generation seed

All judges receive the same:
- blind judge prompt
- taxonomy definitions
- eight scoring dimensions
- Python decision rule
- corresponding candidate-specific numeric seed

Models are loaded sequentially so they do not occupy GPU memory at the same time.

## Scenario assignment

Python creates `scenario_plan.csv` once.

The same matrix row and candidate index therefore use the same scenario assignment for A, B, C, and D.

The LLM never selects or writes scenario metadata.

## Generator output

Each LLM returns only:
- `benchmark_prompt`
- `main_goal`
- `chemical_entity`
- `distinctive_dimension`

Python writes:
- generator identity
- matrix identity
- HC, HD, OT
- allowed scenarios
- selected scenarios
- prompt and scenario versions
- seeds
- attempt count
- timestamp

One candidate is generated per call and saved immediately.

## Blind judging

The judge prompt does not include:
- generator family
- generator model
- candidate ID
- main_goal
- chemical_entity
- distinctive_dimension

The judge receives the controlled taxonomy assignment, scenario assignment, and `benchmark_prompt`.

The model returns eight 1-to-5 scores and `judge_reason`.

It does not return ACCEPT, REVISE, or REJECT.

## Python decision rule

Python derives every judge decision identically:

- REJECT if any core score is 1 or 2.
- ACCEPT if all eight scores are 4 or 5.
- REVISE otherwise.

This prevents contradictory model-generated score/decision combinations.

## Judge health check

V4 monitors score-vector collapse.

If one identical eight-score vector accounts for at least 95% of a rolling window after at least 12 new judgments, that judge run stops instead of silently writing hundreds of suspicious results.

This is specifically designed to catch failures like the earlier uniform-score validator behavior.

## Model comparison

V4 reports two automated generator rankings:

1. All-judge ranking
   - every generator is evaluated by the identical A/B/C/D panel

2. Cross-family ranking
   - the generator's own family judge is excluded
   - useful for measuring sensitivity to self-family scoring

Neither automated ranking is treated as final ground truth.

V4 also measures:
- generation completion
- first-attempt success
- mean scores by dimension
- ACCEPT/REVISE/REJECT rates
- consensus decisions
- self-family judge bias
- generator-by-judge cross matrix
- pairwise decision agreement
- Cohen's kappa
- mean absolute score differences

## Human calibration

Model-model agreement does not establish judge accuracy.

The full pilot creates a blinded human sample with one task per HC category per generator family, normally 36 tasks.

The reviewer scores the same eight dimensions.

Python derives the human reference decision using the same fixed decision rule.

The analysis then creates:
- `judge_vs_human.csv`
- `generator_vs_human.csv`
- `human_review_reference_scored.csv`

This allows the final generator and judges to be selected with human-calibrated evidence.

## Smoke run

Use `run_config_smoke.json` first.

It creates:
- 1 matrix row
- 2 candidate positions
- 4 generators
- 8 candidates
- 32 judgments

The smoke output is kept under `outputs_smoke/`.

## Full V4 pilot

Use `run_config.json` after the smoke test.

It creates:
- 18 matrix rows across HC1-HC9
- 5 candidates per matrix row
- 90 candidates per generator
- 360 total generated candidates
- 1,440 total judgments

The full-pilot output is kept under `outputs/`.

## GitHub and persistence

Every accepted candidate and judgment is written immediately to the Colab clone.

Those files are temporary until downloaded or pushed.

V4 includes optional periodic GitHub checkpointing. A GitHub personal access token is entered only at runtime and is not stored in the repository.

## Reproducibility

V4 records:
- experiment ID
- Git commit
- matrix hash
- taxonomy hash
- prompt hashes
- model-registry hash
- package versions
- GPU information
- prompt versions
- scenario-plan version
- seeds

After a successful run, `environment_freeze.txt` stores the Python version and `pip freeze`.

## Important limitation

No multi-model LLM pipeline can be guaranteed to work perfectly.

V4 is designed to make generation and judging failures visible, reproducible, resumable, and comparable before ChemBreak is scaled to thousands of tasks.


## Interpretation of the model comparison

The labels A-D identify specific checkpoints, not entire model families.

V4 should be reported as a comparison of the four registered checkpoints under
the same local 4-bit inference policy.

If one checkpoint wins, the defensible conclusion is that it performed best
among the tested representatives under this ChemBreak configuration.

## V4.2 Colab runtime fix

V4.2 does not upgrade Colab's preinstalled `torch`, `torchvision`, or `torchaudio`.

The notebook checks for the `_chunk_or_narrow_cat` PyTorch consistency marker before importing the ChemBreak model runtime. If the active kernel contains a mixed PyTorch installation, it tells the user to restart the Colab session before continuing.

This change affects environment setup only. The four-family experimental methodology, generator prompt, judge prompt, matrix, taxonomy, scenario controller, model registry, scoring rule, and comparison design are unchanged.


## V4.2 environment bootstrap

V4.2 fixes the Colab PyTorch inconsistency at the source.

The notebook does not import PyTorch before setup. `bootstrap_colab.py` probes
the existing stack in a separate Python process. The exact healthy target is:

- torch 2.11.0
- torchvision 0.26.0
- torchaudio 2.11.0
- CUDA 12.8 wheels

If that exact stack is already healthy, V4.2 keeps it. Otherwise it performs
one forced reinstall from the official PyTorch CUDA 12.8 index and verifies
the result in a child process.

Only after that succeeds are Transformers, Accelerate, bitsandbytes and the
ChemBreak model runtime imported into the notebook kernel.

This bootstrap change does not alter the four-family research design.
