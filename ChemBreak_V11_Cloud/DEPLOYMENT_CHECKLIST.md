
# Enterprise-first deployment note

For the current deployment, use `ChemBreak_V11_Colab_Enterprise.ipynb` as the primary notebook.

- Do not mount Google Drive.
- Use Colab Enterprise ADC.
- Use Cloud Storage for durable run checkpoints.
- Start with `RUN_TYPE = "test"`.
- A CPU-only runtime is sufficient for controller work.
- The standard Google Colab notebook remains in the package only for portability and is not the recommended deployment path.

# ChemBreak V11 Deployment Checklist

Before running the 9-task test:

- [ ] Folder is named exactly `ChemBreak_V11_Cloud` in GitHub.
- [ ] Open `ChemBreak_V11_Google_Colab.ipynb`.
- [ ] V11 verification gate reports all checks `OK`.
- [ ] `RUN_TYPE` is `test`.
- [ ] Preflight reports `OK` for generator, repair_model, judge_a, judge_b, and adjudicator.
- [ ] Plan contains 9 test assignments and covers HC1 through HC9.
- [ ] Drive output path is under `MyDrive/ChemBreak_V11/outputs/test/`.
- [ ] Generation cell visibly prints progress bars and heartbeat messages.
- [ ] Judge stage shows Judge A and Judge B calls overlapping in time when both are missing.
- [ ] Judge technical errors do not create a selection.
- [ ] Final output and checkpoint ZIP are reviewed before moving to pilot.
