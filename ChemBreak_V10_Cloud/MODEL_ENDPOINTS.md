# ChemBreak V10 Vertex AI model endpoints

These endpoint settings were confirmed by live preflight in the project before the V10 package was created.

| Role | Model selector | Region | API |
| --- | --- | --- | --- |
| Primary generator | `gemini-3.1-pro-preview` | `global` | Gemini generateContent |
| Diversity generator | `meta/llama-4-maverick-17b-128e-instruct-maas` | `us-east5` | OpenAI-compatible `v1beta1` |
| Generator / repair / reasoning judge | `openai/gpt-oss-120b-maas` | `global` | OpenAI-compatible `v1` |
| Stable judge | `gemini-2.5-pro` | `global` | Gemini generateContent |
| Adjudicator | `gemini-3.1-pro-preview` | `global` | Gemini generateContent |

Llama 4 Maverick requires the `Llama 4 API Service` model to be enabled in Model Garden for the Google Cloud project.
