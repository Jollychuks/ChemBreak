# ChemBreak V11 Vertex AI endpoints

| Role | Model | Location | API style |
|---|---|---|---|
| generator | `gemini-3.1-pro-preview` | global | Gemini generateContent |
| repair_model | `gemini-3.1-pro-preview` | global | Gemini generateContent |
| judge_a | `openai/gpt-oss-120b-maas` | global | OpenAI-compatible `v1` |
| judge_b | `gemini-2.5-pro` | global | Gemini generateContent |
| adjudicator | `gemini-3.1-pro-preview` | global | Gemini generateContent |

Generation, repair, refill, and adjudication share the same Gemini 3.1 Pro endpoint, so preflight deduplicates repeated endpoint checks where possible.

V11 has no Llama endpoint.
