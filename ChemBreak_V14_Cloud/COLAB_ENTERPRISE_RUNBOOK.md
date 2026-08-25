# ChemBreak V14 Colab Enterprise Runbook

## Primary notebook

Use `ChemBreak_V14_Colab_Enterprise.ipynb`.

## Runtime

A CPU runtime is sufficient because model inference is remote. V14 uses `/content/chembreak_v14_runtime` for working files.

## Durable storage

V14 writes checkpoints to:

`gs://rs-foundsecft-mghasemi-default-1/ChemBreak_V14/outputs/<run_type>/`

The notebook checks object read/write/delete access and verifies that a temporary `.csv` object is stored with `text/csv` metadata.

## Console behavior

V14 defaults to clean console mode. You will normally see stage start/completion, compact progress, retries, fallbacks, failures, and long waits. Full call-level diagnostics are stored in `technical_events.jsonl`.

## Run order

1. Configure project/repository.
2. Verify bucket access and CSV MIME metadata.
3. Select `RUN_TYPE = "test"` and restore any V14 checkpoint.
4. Initialize the stage runner.
5. Preflight.
6. Bootstrap.
7. Plan.
8. Generate.
9. Validate.
10. Repair.
11. Pre-judge recovery.
12. Judge.
13. Adjudicate.
14. Recovery cycles only while target is incomplete.
15. Finalize.
16. Status and final sync.

## Generation behavior

Normal path: one Gemini 3.1 batch call returns A/B/C.

If batch JSON is malformed:

- retry the batch once;
- if still malformed, fall back to smaller structured calls for the missing slots;
- checkpoint each successful fallback candidate immediately.

## Resume

After a runtime restart, rerun Sections 1-4. Section 3 restores the V14 checkpoint from Cloud Storage. Rerun the interrupted stage. Completed units are skipped by the checkpoint/run-signature guard.
