# Model source notes

These notes document the default model IDs used by V3. The Colab preflight is the final authority on what is enabled in the Google Cloud project.

## Attacker

`gemini-3.1-pro-preview`

This is the confirmed Gemini Pro Preview endpoint for the V3 attacker. It is used for frozen C1/C2 asset generation, C3 graph construction, and live response-adaptive C3 decisions.

## Safety judge

`openai/gpt-oss-120b-maas`

The package uses the OpenAI-compatible Vertex MaaS endpoint, compact JSON output, bounded retries, and adaptive pacing for HTTP 429 responses.

## Chemistry judge

`gemini-2.5-pro`

The model is used only to judge chemical relevance, validity, plausibility, internal consistency, representation accuracy, and requested-output fulfillment. V3 uses an explicit JSON response schema and a bounded thinking budget for this role.

## Adjudicator

`meta/llama-4-maverick-17b-128e-instruct-maas`

Llama 4 Maverick MaaS is configured in `us-east5` and must be enabled for the project.

## Target chemistry models

- ChemDFM: `OpenDFM/ChemDFM-v1.5-8B`
- ChemLLM: `AI4Chem/ChemLLM-7B-Chat-1_5-SFT`
- LlaSMol adapter: `osunlp/LlaSMol-Mistral-7B`
- LlaSMol base: `mistralai/Mistral-7B-v0.1`
