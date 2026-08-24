# ChemBreak V8 Standard Google Colab Runbook

This notebook runs the same independent `ChemBreak_V8_Cloud` V8 codebase from ordinary Google Colab.

## Requirements

- Sign into Colab with a Google account that has access to the Google Cloud project `rs-foundsecft-mghasemi`.
- Commit the complete `ChemBreak_V8_Cloud` folder to the GitHub repository.
- Vertex AI access must be enabled for the configured models.

## Important architecture

Ordinary Colab is only the controller. The configured language models run through Vertex AI, so an A100 Colab runtime is not required.

## Run order

1. Authenticate to Google Cloud.
2. Mount Google Drive.
3. Clone GitHub.
4. Install requirements.
5. Keep `RUN_TYPE = "test"`.
6. Run preflight.
7. Bootstrap.
8. Plan.
9. Inspect coverage.
10. Generate.
11. Validate.
12. Repair.
13. Judge.
14. Refill.
15. Judge again.
16. Finalize.
17. Create a checkpoint ZIP.

## Persistence

Outputs are stored under:

`MyDrive/ChemBreak_V8/outputs/<run_type>/`

This allows test, pilot, and production to resume separately after a Colab runtime reset.

## Independence from V7

Only the GitHub folder `ChemBreak_V8_Cloud/` is used. No V7 code or output is imported.
