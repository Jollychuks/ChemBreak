# ChemBreak V12 Colab Enterprise Runbook

## Deployment target

Use `ChemBreak_V12_Colab_Enterprise.ipynb` in Google Cloud Colab Enterprise. No standard Google Colab notebook is included in this release.

## Runtime

A CPU-only runtime is sufficient because LLM inference is performed remotely by Vertex AI. The Enterprise runtime mainly performs orchestration, deterministic validation, CSV/JSON work, network calls, and checkpoint synchronization.

## Authentication

Use a Colab Enterprise runtime with end-user credentials. The notebook uses Application Default Credentials.

## Durable storage

The notebook uses the existing bucket:

`gs://rs-foundsecft-mghasemi-default-1/`

V12 writes under:

`gs://rs-foundsecft-mghasemi-default-1/ChemBreak_V12/outputs/<run_type>/`

It does not attempt to create a bucket. Section 2 performs an object-level write/read/delete probe before the experiment starts.

Changed output files are mirrored every 60 seconds while a stage is running and once more when a stage ends or errors.

## Run order

1. Configuration, ADC, GitHub clone, dependencies.
2. Verify existing bucket access.
3. Set `RUN_TYPE = "test"` and restore any V12 checkpoint.
4. Initialize the live stage runner.
5. Preflight.
6. Bootstrap.
7. Plan.
8. Generate.
9. Validate.
10. Repair.
11. Pre-judge refill.
12. Two independent judges.
13. Blind adjudication.
14. Full refill cycles.
15. Finalize.
16. Status and final sync.

Run the 9-task test before pilot and production.

## V12 speed changes

The initial generation stage normally makes one Gemini call per assignment and returns A/B/C together. In V11 this stage required three calls per assignment. Judge B now receives an explicit response schema, preventing the repeated malformed-JSON retry pattern observed in V11.

## Resume

After a runtime restart, rerun Sections 1-4. Section 3 restores the durable Cloud Storage checkpoint. Then rerun the interrupted stage. The run signature and checkpoint files determine what is already complete.
