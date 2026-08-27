# Colab Enterprise setup

Repository: `https://github.com/Jollychuks/ChemBreak`

Upload this folder to:

`ChemBreak_Adaptive_Jailbreak_v1_1_1/`

## Before the first run

1. Confirm the Vertex AI / Agent Platform API is enabled in your project.
2. Confirm `openai/gpt-oss-120b-maas` works in the project.
3. In Model Garden, enable Llama 4 Maverick and accept the EULA. Maverick MaaS uses `us-east5`.
4. Confirm Gemini 2.5 Pro is available.
5. For the attacker, the notebook defaults to the confirmed working `gemini-3.1-pro-preview`. If your console exposes a distinct Gemini 3.5 Pro Preview ID, paste that exact ID into the configuration cell.
6. Use an A100 runtime for ChemDFM, ChemLLM, and LlaSMol.

## Recommended run order

1. Clone/refresh GitHub repository.
2. Install dependencies.
3. Configure TEST and a stable `RUN_ID`.
4. Locate the V15 final task bank in private GCS.
5. Create `configs/runtime.yaml`.
6. Run preflight.
7. Import completed v1.1 frozen assets if a compatible prior run exists (the notebook does this safely and is a no-op otherwise).
8. Run `prepare`.
9. Execute ChemDFM.
10. Execute ChemLLM.
11. Execute LlaSMol.
12. Rebuild metrics.
13. Inspect test outputs.
14. Repeat with PILOT.
15. Freeze the methodology.
16. Run PRODUCTION.

The notebook never pushes to GitHub.
