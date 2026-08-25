# ChemBreak V15 Colab Enterprise Runbook

Use `ChemBreak_V15_Colab_Enterprise.ipynb`. A CPU runtime is sufficient because model inference is remote. V15 uses `/content/chembreak_v15_runtime` for working files and writes durable checkpoints to `gs://rs-foundsecft-mghasemi-default-1/ChemBreak_V15/outputs/<run_type>/`.

## Test sequence

1. Configure project/repository and verify the existing bucket.
2. Keep `RUN_TYPE = "test"`.
3. Run preflight and require all model roles to show `OK`.
4. Confirm Judge A's contract test passes the 0-5 scoring rule.
5. Run bootstrap, plan, generate, validate, repair, pre-judge refill, judge, adjudicate, recovery cycles, finalize, and status.

V15 defaults to clean console output: stage start/completion, compact progress, retries, fallbacks, failures, long waits, and summaries. Full call-level diagnostics are retained in `technical_events.jsonl` and `errors.jsonl`.

After a runtime restart, rerun the setup/restore sections and rerun the interrupted stage. Completed units are skipped by the checkpoint and run-signature guards.
