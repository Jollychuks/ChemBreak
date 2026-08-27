# Model source notes

These notes document the default model IDs used by v2. The Colab preflight is the final authority on what is enabled in the user's Google Cloud project.

## Attacker

Default: `gemini-3.1-pro-preview`

Reason: this exact endpoint was already used successfully in the user's prior ChemBreak Google Cloud run. Current Google Cloud documentation lists Gemini 3.1 Pro preview as a supported Pro model. If the user's console exposes a separate Gemini 3.5 Pro Preview model ID, the notebook supports overriding the attacker ID without code changes.

## Safety judge

`openai/gpt-oss-120b-maas`

Google Cloud documents gpt-oss 120B as a reasoning-oriented MaaS model. The package uses the OpenAI-compatible Vertex endpoint and JSON response mode.

Documentation:
https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/maas/openai/gpt-oss-120b

## Chemistry judge

`gemini-2.5-pro`

The model is used only to judge chemical relevance, validity, plausibility, internal consistency, representation accuracy, and requested-output fulfillment.

## Adjudicator

`meta/llama-4-maverick-17b-128e-instruct-maas`

Llama 4 Maverick MaaS is served from `us-east5` and requires Model Garden enablement plus EULA acceptance.

Documentation:
https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/partner-models/llama/llama4-maverick

## Target chemistry models

- ChemDFM: `OpenDFM/ChemDFM-v1.5-8B`
- ChemLLM: `AI4Chem/ChemLLM-7B-Chat-1_5-SFT`
- LlaSMol adapter: `osunlp/LlaSMol-Mistral-7B`
- LlaSMol base: `mistralai/Mistral-7B-v0.1`
