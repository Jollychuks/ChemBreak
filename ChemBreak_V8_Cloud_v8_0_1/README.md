# ChemBreak V8.0.1 Cloud

This is the GitHub-ready, standalone Vertex AI version of the ChemBreak task-bank generator.

## Independence from V7

V8 does not call, import, or read any V7 code or V7 generated task files.

The Colab Enterprise notebook clones the GitHub repository and then uses only:

`ChemBreak_V8_Cloud/`

Therefore, after this folder is safely committed to GitHub, deleting a separate V7 folder will not affect V8.

## Repository structure

- `ChemBreak_V8_Cloud_Colab_Enterprise.ipynb`  
  Thin Colab Enterprise controller that clones the repository and runs this V8 folder.
- `scripts/chembreak_v8_cloud.py`  
  Main generation, validation, repair, judging, refill, and finalization pipeline.
- `prompts/`  
  Every generator, repair, judge, adjudicator, and refill prompt as a separate editable text file.
- `taxonomy/`  
  The full V8 taxonomy plus CSV views of HC, HD, OT, scenario, HC-HD, and HC-OT mappings.
- `config/run_config.json`  
  Vertex model roles, run sizes, thresholds, project ID, and generation settings.
- `data/source_manifest.json`  
  Upstream data provenance. Bootstrap creates fresh source snapshots in the run output.
- `outputs/`  
  Git-ignored placeholder. Production results should normally live in GCS or downloadable checkpoints.

## Default model roles

- Primary generator: Gemini 3.1 Pro Preview
- Diversity generator: Llama 4 Maverick
- Additional generator: gpt-oss-120B
- Repair: gpt-oss-120B
- Judge 1: Gemini 2.5 Pro
- Judge 2: gpt-oss-120B
- Adjudicator: Gemini 3.1 Pro Preview

Mistral Large 3 has a disabled config slot. Enable it only after copying the exact current model selector and region from its Vertex Model Garden **View Code** panel.

## Recommended workflow

1. Upload the whole `ChemBreak_V8_Cloud` folder to the root of the GitHub repository.
2. In Google Cloud, open Colab Enterprise.
3. Import `ChemBreak_V8_Cloud_Colab_Enterprise.ipynb`.
4. Connect to a normal runtime. A notebook GPU is not required for the serverless Vertex models.
5. Run `preflight`.
6. Run `bootstrap`.
7. Run `plan`.
8. Inspect matrix coverage.
9. Run `generate`.
10. Run `validate`.
11. Run `repair`.
12. Run `judge`.
13. Run `refill`.
14. Run `judge` again.
15. Run `finalize`.

Always complete `test` first, then `pilot`, then `production`.

## Run sizes

- Test: 9 final targets
- Pilot: 100 final targets plus 15 reserve assignments
- Production: 500 final targets plus 75 reserve assignments

With the three default generators, production begins with up to 1,725 raw candidate attempts before repairs/refills.

## Prompts

The prompt text is no longer hidden inside the notebook. It is in `prompts/` and is loaded by the pipeline at runtime. Edit those files in GitHub whenever you intentionally create a new prompt version.

## Persistence

For production, set `GCS_OUTPUT_URI` in the notebook. This lets V8 pull and push checkpoints so a deleted Colab Enterprise runtime does not erase the experiment.

## Important

The generated benchmark request is the artifact. The generator and repair prompts explicitly tell models not to solve the underlying chemistry request.


## Standard Google Colab

Use `ChemBreak_V8_Google_Colab.ipynb` to run V8 from ordinary Google Colab.

The notebook authenticates Colab to the Google Cloud project, mounts Google Drive for persistent checkpoints, clones this GitHub repository, and uses this same V8 codebase. It does not depend on V7.


See `PATCH_NOTES_V8_0_1.md` for the live Vertex preflight fixes.
