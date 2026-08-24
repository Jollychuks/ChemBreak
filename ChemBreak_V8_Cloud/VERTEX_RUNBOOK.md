# Vertex AI / Colab Enterprise Runbook

1. Put the entire `ChemBreak_V8_Cloud` folder in your GitHub repository.
2. Confirm the folder path is exactly `ChemBreak_V8_Cloud`.
3. Import `ChemBreak_V8_Cloud_Colab_Enterprise.ipynb` into Colab Enterprise.
4. Run the clone cell.
5. Confirm the printed V8 path points to `/tmp/ChemBreak_repo/ChemBreak_V8_Cloud`.
6. Run the install cell.
7. Keep `RUN_TYPE = "test"`.
8. Run preflight and inspect `preflight_models.csv`.
9. If a required model reports 403 or permission denied, stop and fix model access first.
10. Bootstrap source data.
11. Build and inspect the fresh V8 plan.
12. Run generation through finalization stage by stage.
13. Download the checkpoint ZIP or set a GCS path before the runtime is deleted.
14. After the test looks correct, change to `pilot` and repeat in its fresh output directory.
15. Run production only after pilot review and institutional approval for production-scale Vertex usage.
