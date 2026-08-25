# ChemBreak V13 Deployment Checklist

- [ ] Upload `ChemBreak_V13_Cloud/` as a new GitHub folder.
- [ ] Leave V9-V12 folders untouched.
- [ ] Open `ChemBreak_V13_Colab_Enterprise.ipynb`.
- [ ] Connect a Colab Enterprise runtime.
- [ ] Confirm ADC and project setup succeed.
- [ ] Confirm existing bucket access probe succeeds.
- [ ] Confirm working root is `/content/chembreak_v13_runtime`.
- [ ] Keep `RUN_TYPE = "test"` for the first run.
- [ ] Confirm preflight shows all roles `OK`.
- [ ] Confirm Judge A contract preflight is `OK`.
- [ ] Confirm Judge B contract preflight is `OK`.
- [ ] Run the 9-task test through finalize.
- [ ] Review completion, judge agreement, technical failures, refill rate, and diversity before pilot.
- [ ] Run the 100-task pilot before 500-task production.
