# Task bank input

Place the completed ChemBreak `final_task_bank.csv` here as `final_task_bank.csv`, or point `input.task_bank_uri` in `configs/gcp.yaml` to its `gs://` location.

The runner auto-detects common V15-style column names for task ID, benchmark prompt, HC category, hazard domain, output type, entity, and scenario. It does not invent missing taxonomy mappings.
