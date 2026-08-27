# Task bank input

Point `input.task_bank_uri` to the completed ChemBreak V15 `final_task_bank.csv` in restricted GCS.

The runner auto-detects common V15-style columns for task ID, benchmark prompt, HC category, hazard domain, output type, entity, and scenario. It also detects optional SMILES, formula, and IUPAC/systematic-name columns when present. Missing optional chemistry fields are left empty. The pipeline never invents missing taxonomy or chemistry metadata.
