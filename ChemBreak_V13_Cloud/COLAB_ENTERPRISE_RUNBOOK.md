# ChemBreak V13 Colab Enterprise Runbook

## Primary notebook

Use `ChemBreak_V13_Colab_Enterprise.ipynb`.

## Runtime

A CPU runtime is sufficient for the controller. The model inference is remote. If an A100 runtime is already connected, V13 will still work, but the GPU is not required.

V13 uses `/content/chembreak_v13_runtime` for working files so the larger content disk is used rather than the smaller root disk.

## Durable checkpoint storage

V13 uses the existing bucket:

`gs://rs-foundsecft-mghasemi-default-1/ChemBreak_V13/`

The notebook performs a write/read/delete access probe before the run.

## Run order

1. Configure project, repository, and credentials.
2. Verify bucket access.
3. Select `RUN_TYPE = "test"` and restore any V13 checkpoint.
4. Initialize the live stage runner.
5. Preflight.
6. Bootstrap.
7. Plan.
8. Generate.
9. Validate.
10. Repair.
11. Pre-judge refill.
12. Judge.
13. Adjudicate.
14. Full refill cycles as needed.
15. Finalize.
16. Status.

## V13 preflight

Preflight now checks endpoint reachability and performs a harmless serialization smoke test of the actual judge contract for both Judge A and Judge B. Do not proceed if either judge contract is `ERROR`.

## Resume

After a runtime restart, rerun the setup and restore cells. Then rerun the interrupted stage. V13 checkpoints completed units and does not intentionally regenerate completed work.
