# ChemBreak V11 - Colab Enterprise Runbook

## Deployment target

Use `ChemBreak_V11_Colab_Enterprise.ipynb` in Google Cloud Colab Enterprise.

The V11 task-bank logic is unchanged. This deployment variant changes only orchestration, authentication, and checkpoint persistence.

## Runtime recommendation

A CPU-only Colab Enterprise runtime is sufficient for the ChemBreak controller because Gemini 3.1 Pro, gpt-oss-120B, and Gemini 2.5 Pro inference is executed by managed Vertex AI endpoints. The notebook runtime mainly performs Python orchestration, validation, CSV/JSON processing, checkpointing, and network calls.

## Authentication

Prefer a Colab Enterprise runtime template with end-user credentials enabled. The notebook uses Application Default Credentials and does not use the standard Google Colab authentication flow.

## Durable storage

Runtime-local files are not the source of truth. The Enterprise notebook mirrors the run output to:

`gs://<bucket>/ChemBreak_V11/outputs/<run_type>/`

The default bucket name is:

`<project-id>-chembreak-v11`

If bucket creation is blocked by institutional IAM, set `GCS_BUCKET` in Section 1 to an existing bucket where you have object read/write permission.

Changed checkpoint files are mirrored every 60 seconds while each stage is running and once more when the stage ends or raises an error.

## Run order

1. Configuration, ADC, GitHub clone, dependencies.
2. Cloud Storage setup.
3. Select `RUN_TYPE = "test"` and restore checkpoint.
4. Initialize the stage runner.
5. Preflight.
6. Bootstrap.
7. Plan.
8. Generate.
9. Validate.
10. Repair.
11. Pre-judge refill.
12. Concurrent judging.
13. Adjudication.
14. Full refill cycles.
15. Finalize.
16. Status.

Run the 9-task test before pilot and production.

## Resume

If a runtime is restarted or deleted:

1. Connect to a new Colab Enterprise runtime.
2. Rerun Sections 1-4.
3. Section 3 restores the durable checkpoint from Cloud Storage.
4. Rerun the interrupted stage.
5. Continue normally.

The V11 run signature and compatibility guard prevent an incompatible pipeline/configuration from silently mixing with an existing checkpoint.

## Production execution

After the interactive 9-task test and 100-task pilot succeed, you can also use a Colab Enterprise one-off notebook execution with a runtime template and Cloud Storage output location if you want the notebook to run independently of your browser session.
