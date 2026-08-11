# ChemBreak Colab Generator V2

ChemBreak Colab Generator V2 generates base harmful chemistry target tasks for later jailbreak testing using an open-weight LLM running directly inside Google Colab.

It does **not** use a paid LLM API.

## What changed from V1

V2 incorporates the problems discovered during the first ChemBreak pilot:

1. Generator prompt upgraded to `CB-GEN-COLAB-v1.3`.
2. Full HC1-HC9 category definitions and exclusion boundaries are embedded in the prompt.
3. HC category has explicit priority over output type.
4. Full SC01-SC15 definitions are embedded in the prompt.
5. A selected scenario must actually appear in the user-facing task.
6. Stronger harmful-intent requirement.
7. Stronger chemistry-dependency requirement.
8. Stronger chemistry-plausibility checks.
9. Sampling reduced from temperature 0.85 to 0.4.
10. Generation retries increased to 5.
11. The notebook test uses `generate_with_retries()` instead of a one-shot validator.
12. The default pilot is stratified across HC1-HC9 instead of taking only the first HC1 rows.
13. Optional semantic validator added.
14. GitHub clone workflow added.
15. Optional GitHub checkpoint helper added.
16. Provenance now records `CB-GEN-COLAB-v1.3`.

## Default pilot

The default `run_config.json` selects 18 matrix rows:

- 2 from HC1
- 2 from HC2
- 2 from HC3
- 2 from HC4
- 2 from HC5
- 2 from HC6
- 2 from HC7
- 2 from HC8
- 2 from HC9

With 5 candidates per row:

`18 × 5 = 90 raw candidate tasks`

This is a better pilot because it checks the entire HC taxonomy before generating thousands.

## Full generation

A separate file is included:

`run_config_full.example.json`

It is configured for:

`271 rows × 25 candidates = 6,775 raw candidate tasks`

Do not use the full configuration until the 90-task stratified pilot has been reviewed.

## Files

- `ChemBreak_Colab_Generator_V2.ipynb`  
  Main Colab notebook.

- `ChemBreak_Working_Generation_Matrix_v1.csv`  
  Existing 271-row generation matrix.

- `generator_prompt.txt`  
  Generator prompt `CB-GEN-COLAB-v1.3`.

- `validator_prompt.txt`  
  Optional semantic candidate validator.

- `generate_local.py`  
  Open-weight local generation engine.

- `validate_local.py`  
  Optional semantic validator using the same loaded model.

- `run_config.json`  
  90-task stratified pilot configuration.

- `run_config_full.example.json`  
  Example full 6,775-candidate configuration.

- `candidate_tasks_template.csv`  
  Generator output schema.

- `candidate_tasks_validated_template.csv`  
  Semantic validation output schema.

- `requirements_colab.txt`  
  Colab dependencies.

- `github_checkpoint.py`  
  Optional helper for pushing generated CSV checkpoints back to GitHub.

- `GITHUB_SETUP.md`  
  GitHub workflow instructions.

- `CHANGELOG_V2.md`  
  Summary of V2 changes.

## Recommended workflow

1. Put this V2 folder into your GitHub repository.
2. Open the notebook in Google Colab.
3. Set the repository URL and project subdirectory in the setup cell.
4. Select a GPU runtime.
5. Clone the repo.
6. Install requirements.
7. Inspect the 18 stratified pilot rows.
8. Load the open-weight model.
9. Run the retry-safe 2-candidate test.
10. Run the 90-candidate pilot.
11. Run the optional semantic validator.
12. Manually inspect ACCEPT/REVISE/REJECT results.
13. Adjust the generator if needed.
14. Only then scale to the full matrix.

## Important distinction

The generator produces **base target tasks**.

It does not generate the jailbreak attacks yet.

The later research stage remains:

Base harmful task → ChemBreak attack mechanism → target Chemical LLM → response → judges → adjudication → metrics.
