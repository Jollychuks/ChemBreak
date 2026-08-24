# ChemBreak V11 Standard Google Colab Runbook

1. Upload the complete `ChemBreak_V11_Cloud/` folder to the root of the ChemBreak GitHub repository.
2. Open `ChemBreak_V11_Google_Colab.ipynb` in standard Google Colab.
3. A CPU runtime is sufficient. Vertex AI performs the LLM inference remotely.
4. Authenticate with Application Default Credentials using the notebook's `gcloud auth application-default login --no-launch-browser` cell.
5. Set the quota project to `rs-foundsecft-mghasemi`.
6. Mount Google Drive. V11 writes to `/content/drive/MyDrive/ChemBreak_V11/outputs/<run_type>/` by default.
7. Clone or refresh GitHub and confirm the hard V11 verification gate reports all checks `OK`.
8. Install requirements.
9. Keep `RUN_TYPE = "test"` for the first run.
10. Run preflight. Every required model role must report `OK`.
11. Bootstrap fresh source provenance and build the plan.
12. Inspect HC, HD, OT, and candidate-language-profile coverage.
13. Run generation, validation, repair, pre-judge refill, concurrent judging, and adjudication.
14. Run the recovery loop. It can perform up to three full refill cycles.
15. Finalize and inspect `pipeline_metrics.json`, coverage, diversity, similarity, and the final task bank.
16. Create the checkpoint ZIP.

## Resume behavior

Every durable artifact is written immediately. If Colab terminates, reconnect, rerun setup/authentication/Drive/GitHub/config cells, then rerun the interrupted stage. Existing completed candidate IDs, repair attempts, judge records, outcomes, and selected assignments are reused.

If a judge succeeded and the other judge failed technically, rerunning `judge` reuses the successful record and retries only the missing judge for the same candidate-set hash.

## Progress output

V11 prints a text progress bar, percentage, elapsed time, ETA, current candidate or assignment, model call start/completion messages, and a heartbeat every 20 seconds while waiting for long Vertex calls.
