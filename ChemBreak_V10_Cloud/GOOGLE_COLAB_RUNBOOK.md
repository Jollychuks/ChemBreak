# ChemBreak V10 Google Colab runbook

1. Upload the complete `ChemBreak_V10_Cloud` folder to the root of the GitHub repository.
2. Open `ChemBreak_V10_Google_Colab.ipynb` in standard Google Colab.
3. Run Python setup.
4. Authenticate with Application Default Credentials.
5. Attach `rs-foundsecft-mghasemi` as quota and active project.
6. Mount Google Drive.
7. Clone GitHub.
8. Confirm the path is `/content/ChemBreak_repo/ChemBreak_V10_Cloud`.
9. Install requirements.
10. Keep `RUN_TYPE = "test"`.
11. Run preflight. All required roles must report `OK`.
12. Run bootstrap.
13. Run plan.
14. Review HC, HD, OT, request-form, entity, and scenario coverage before generation.
15. Run generation.
16. Run deterministic validation.
17. Run repair.
18. Run blind judging.
19. Run refill.
20. Run blind judging again.
21. Run finalization.
22. Review coverage and diversity reports.
23. Create the checkpoint ZIP.
24. Repeat with `pilot`.
25. Run `production` only after pilot review and approval for production-scale Vertex usage.
