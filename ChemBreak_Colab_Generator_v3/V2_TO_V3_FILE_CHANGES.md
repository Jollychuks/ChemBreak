# V2 to V3 File Changes

| File | V3 status | Difference from V2 |
|---|---|---|
| `.gitignore` | **UNCHANGED** | Same cache and notebook-ignore behavior. |
| `ChemBreak_Working_Generation_Matrix_v1.csv` | **UNCHANGED** | Same 271 matrix rows. V3 changes generation control, not the matrix. |
| `candidate_tasks_template.csv` | **UNCHANGED** | Same raw candidate output schema. |
| `candidate_tasks_validated_template.csv` | **UNCHANGED** | Same semantic-validation output schema. |
| `github_checkpoint.py` | **UNCHANGED** | Same optional GitHub push helper. |
| `requirements_colab.txt` | **UNCHANGED** | Same Colab dependencies. |
| `validate_local.py` | **UNCHANGED** | Same post-generation semantic validator. |
| `validator_prompt.txt` | **UNCHANGED** | Same semantic validator rubric. |
| `generate_local.py` | **CHANGED** | Python now assigns scenarios, generates one candidate per call, immediately checkpoints each accepted candidate, injects selected taxonomy definitions, rejects exact within-row duplicates, and prevents accidental mixed prompt versions. |
| `generator_prompt.txt` | **CHANGED** | v1.3 to v1.4. Shorter focused prompt. Receives only the selected HC/HD/OT definitions plus the Python-assigned SC instead of the entire taxonomy. |
| `run_config.json` | **CHANGED** | Package 3.0, prompt v1.4, taxonomy file added, end_row null for explicit pilot IDs, no call_batch_size, scenario_pair_rate 0.0, mixed prompt versions disabled. |
| `run_config_full.example.json` | **CHANGED** | Updated to the V3 architecture and V3 provenance while retaining the 271 x 25 full-run target. |
| `ChemBreak_Colab_Generator_V3.ipynb` | **CHANGED** | Replaces V2 notebook. Uses the V3 folder path, previews Python scenario assignments, uses V3 test generation, and launches the V3 one-candidate-at-a-time generator. |
| `README.md` | **CHANGED** | Documents V3 architecture and current output persistence behavior. |
| `QUICK_START.txt` | **CHANGED** | Updated for V3 GitHub folder and Python scenario control. |
| `GITHUB_SETUP.md` | **CHANGED** | Updated V3 paths and explains where generated datasets live. |
| `CHANGELOG_V3.md` | **NEW/RENAMED** | Replaces the V2 changelog with V3-specific changes. |
| `taxonomy_definitions.json` | **NEW** | Central audited definitions for HC1-HC9, HD1-HD8, OT1-OT15, and SC01-SC15. |
| `V2_TO_V3_FILE_CHANGES.md` | **NEW** | File-by-file comparison between V2 and V3. |