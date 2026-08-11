# ChemBreak Colab Generator V3

ChemBreak Colab Generator V3 is the complete standalone Colab package for generating base harmful chemistry target tasks for later ChemBreak jailbreak testing.

V3 does not use a paid LLM API. The open-weight generator model runs directly on the Colab GPU.

## Core V3 architecture

`matrix row -> selected taxonomy definitions -> Python scenario assignment -> one candidate per LLM call -> structural validation -> immediate CSV checkpoint`

The LLM no longer chooses scenario IDs.

## Why V3 replaces V2

The V2 pilot showed that Qwen could repeatedly choose disallowed scenario IDs such as SC04 even when the matrix did not permit SC04. A five-candidate batch could therefore be rejected because one candidate had bad scenario metadata.

V3 removes that failure mode.

Python now:
- reads the allowed scenario list from the matrix
- deterministically assigns zero or one scenario for each pilot candidate
- gives the exact assigned scenario and definition to Qwen
- overwrites selected_scenarios with the Python-controlled value
- generates one candidate per model call
- saves every accepted candidate immediately

## Another V3 improvement

V2 embedded all HC1-HC9 definitions in every generator prompt.

V3 adds `taxonomy_definitions.json`.

For each candidate, Python injects only:
- the selected HC definition, inclusions, and exclusions
- the selected HD definition
- the selected OT definition
- the assigned SC definition

This makes the prompt shorter, more focused, easier to audit, and less likely to confuse a small generator model.

## Default pilot

The V3 pilot selects 18 matrix rows:
- 2 from HC1
- 2 from HC2
- 2 from HC3
- 2 from HC4
- 2 from HC5
- 2 from HC6
- 2 from HC7
- 2 from HC8
- 2 from HC9

It generates 5 candidates per row.

Expected raw pilot size: 90 tasks.

## Files

- `ChemBreak_Colab_Generator_V3.ipynb`
- `ChemBreak_Working_Generation_Matrix_v1.csv`
- `taxonomy_definitions.json`
- `generator_prompt.txt`
- `validator_prompt.txt`
- `generate_local.py`
- `validate_local.py`
- `run_config.json`
- `run_config_full.example.json`
- `candidate_tasks_template.csv`
- `candidate_tasks_validated_template.csv`
- `github_checkpoint.py`
- `requirements_colab.txt`
- `GITHUB_SETUP.md`
- `QUICK_START.txt`
- `CHANGELOG_V3.md`
- `V2_TO_V3_FILE_CHANGES.md`
- `.gitignore`

## Output files created during a run

Generation creates:
- `candidate_tasks.csv`
- `generation_progress.csv`
- `generation_errors.jsonl`

Semantic validation creates:
- `candidate_tasks_validated.csv`
- `validation_progress.csv`
- `validation_errors.jsonl`

These files are first stored inside the cloned Colab copy of the repo. They are not automatically pushed back to GitHub.

Use the optional GitHub checkpoint cell to push them, or download them before deleting the runtime.

## Full-generation example

`run_config_full.example.json` targets:

271 matrix rows x 25 candidates = 6,775 raw candidates.

Do not start the full run until the V3 stratified pilot has been reviewed.
