# ChemBreak V9 Google Colab Runbook

1. Upload the complete `ChemBreak_V9_Cloud` folder to the root of the GitHub repository.
2. Open `ChemBreak_V9_Google_Colab.ipynb` in standard Google Colab.
3. Run Python setup.
4. Authenticate with Application Default Credentials.
5. Set the quota and active project.
6. Mount Google Drive.
7. Clone GitHub.
8. Confirm the printed folder is `/content/ChemBreak_repo/ChemBreak_V9_Cloud`.
9. Install requirements.
10. Keep `RUN_TYPE = "test"`.
11. Run preflight.
12. Continue only when the required model endpoints report `OK`.
13. Run bootstrap, plan, generation, validation, repair, judging, refill, judging again, and finalization.
