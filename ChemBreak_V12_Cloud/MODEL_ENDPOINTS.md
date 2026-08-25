# ChemBreak V12 Vertex AI Model Endpoints

| Role | Model | Location | API style |
|---|---|---|---|
| Generator / Repair / Refill / Adjudicator | `gemini-3.1-pro-preview` | `global` | Vertex Gemini generateContent |
| Judge A | `openai/gpt-oss-120b-maas` | `global` | Vertex OpenAI-compatible `v1` |
| Judge B | `gemini-2.5-pro` | `global` | Vertex Gemini generateContent |

Judge B uses `responseMimeType: application/json` plus an explicit Vertex AI `responseSchema` in V12. The generator batch and other Gemini structured calls also use response schemas where applicable.
