# ChemBreak V14 Deployment Checklist

- [ ] Upload `ChemBreak_V14_Cloud/` as a new GitHub folder.
- [ ] Leave V13 and earlier folders untouched.
- [ ] Open `ChemBreak_V14_Colab_Enterprise.ipynb`.
- [ ] Connect a Colab Enterprise runtime.
- [ ] Confirm working root is `/content/chembreak_v14_runtime`.
- [ ] Confirm bucket access probe succeeds.
- [ ] Confirm `CSV metadata: PASSED (text/csv)`.
- [ ] Keep `RUN_TYPE = "test"` for the first run.
- [ ] Confirm preflight shows generator, Judge A, Judge B, and adjudicator `OK`.
- [ ] Confirm clean console output is readable and not call-by-call noisy.
- [ ] Confirm initial generation reaches 9/9 assignments or uses automatic fallback for any failed batch.
- [ ] Confirm no known entity is assigned to an incompatible hazard domain.
- [ ] Confirm 18 judgment rows for a complete 9-assignment test.
- [ ] Confirm final task bank is `COMPLETE_9_OF_9`.
- [ ] Confirm `.csv` objects in GCS show `text/csv` metadata.
- [ ] Review `technical_events.jsonl`, `errors.jsonl`, judge agreement, refill rate, and diversity.
- [ ] Run the 100-task pilot before 500-task production.
