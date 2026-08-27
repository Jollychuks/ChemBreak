# Colab Enterprise workflow

This project is meant to live as one folder in:

`https://github.com/Jollychuks/ChemBreak`

Recommended repository path:

`ChemBreak_Adaptive_Jailbreak_v1/`

The Colab Enterprise notebook clones the repository into the runtime, changes into this project subdirectory, installs the package, creates a non-committed `configs/runtime.yaml`, performs preflight checks, and then exposes separate cells for prepare, controlled, adaptive, and metrics stages.

## Data handling

The GitHub repository is public. Do not commit the completed harmful production task bank or raw execution transcripts. Keep those in restricted Google Cloud Storage. The supplied `.gitignore` excludes `data/final_task_bank.csv`, `outputs/`, and `configs/runtime.yaml`.

## Recommended sequence

1. Upload this whole folder to the root of the ChemBreak GitHub repository.
2. Commit and push it yourself.
3. Open `ChemBreak_Adaptive_Jailbreak_v1_Colab_Enterprise.ipynb` in Colab Enterprise.
4. Run the clone/setup cell.
5. Confirm `GCP_PROJECT`, `TASK_BANK_URI`, and `GCS_BASE_URI` in the configuration cell.
6. Create `configs/runtime.yaml` from the notebook.
7. Run preflight.
8. Run TEST.
9. Run PILOT and freeze methodology.
10. Change to PRODUCTION only after the pilot settings are frozen.

The notebook never pushes to GitHub.
