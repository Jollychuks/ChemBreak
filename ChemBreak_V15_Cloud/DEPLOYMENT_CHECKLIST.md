# ChemBreak V15 Deployment Checklist

- [ ] Upload `ChemBreak_V15_Cloud/` as a new GitHub folder beside earlier versions.
- [ ] Open `ChemBreak_V15_Colab_Enterprise.ipynb`.
- [ ] Keep `RUN_TYPE = "test"` for the first run.
- [ ] Confirm bucket preflight reports CSV metadata as `text/csv`.
- [ ] Confirm model preflight reports both `judge_a` and `judge_b` as `OK`.
- [ ] In particular, Judge A contract smoke test must pass the 0-5 score validator.
- [ ] Run the 9-task test through finalize/status before pilot.
- [ ] Keep V14 and earlier checkpoints untouched.
