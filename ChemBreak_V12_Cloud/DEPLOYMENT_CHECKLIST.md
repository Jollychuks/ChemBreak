# ChemBreak V12 Deployment Checklist

- Upload `ChemBreak_V12_Cloud/` to the root of the ChemBreak GitHub repository.
- Open `ChemBreak_V12_Colab_Enterprise.ipynb` in Colab Enterprise.
- Use a CPU runtime unless a separate local GPU task is added later.
- Confirm ADC succeeds.
- Confirm the notebook reports `V12 pipeline verification: PASSED`.
- Confirm object access to `gs://rs-foundsecft-mghasemi-default-1/ChemBreak_V12/` passes.
- Keep `RUN_TYPE = "test"` for the first run.
- Confirm all four model roles pass preflight.
- Confirm generation reports one batch call per assignment and saves A/B/C.
- Confirm Judge A and Judge B both produce successful rows in `judgments.csv`.
- Confirm Judge B no longer produces repeated structured-JSON failures.
- Confirm `judge_outcomes.csv` is written.
- Run adjudication/refill only when required by the recorded outcomes.
- Finalize only after selected tasks exist.
- Review `final_task_bank.csv`, coverage, diversity, and pipeline metrics before pilot.
