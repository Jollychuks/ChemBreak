# ChemBreak V10 live-progress update

This update is intended to change observability only, not benchmark-generation behavior.

## Files changed for the live-progress update

- `scripts/chembreak_v10_cloud.py`
- `README.md`
- `RELEASE_NOTES_V10.md`
- this note

## What was added

- immediate `flush=True` output
- stage start and finish messages
- completed/total counts
- percentages
- elapsed time
- ETA
- model-call start and finish messages
- a heartbeat about every 20 seconds during long Vertex AI calls
- visible validation PASS/FAIL status
- visible repair status
- visible judge decisions and adjudication status
- visible refill and finalization progress

## What was not changed

- generator prompts
- repair prompts
- judge prompts
- adjudicator prompts
- taxonomy
- HC/HD/OT/scenario mappings
- model selection
- model endpoint configuration
- temperatures
- generation token budgets
- validation thresholds
- candidate CSV schema
- judgment CSV schema
- assignment planning rules
- candidate selection rules
- deterministic validation logic
- repair acceptance logic
- judge qualification rubric
- adjudication decision rules
- refill decision rules
- final task-bank target sizes

## Metadata correction

The pipeline now reports `VERSION = "10.0-cloud"`. The first V10 package accidentally retained an internal `9.0-cloud` version string. This correction affects only V10 metadata/run signatures and does not change prompts, model calls, validation, or selection logic.
