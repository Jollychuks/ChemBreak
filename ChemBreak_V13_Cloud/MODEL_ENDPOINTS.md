# ChemBreak V13 Model Endpoints

| Role | Model | Location | Notes |
|---|---|---|---|
| Generator | Gemini 3.1 Pro Preview | global | one batch call returns A/B/C |
| Repair / refill | Gemini 3.1 Pro Preview | global | exact-defect recovery |
| Judge A | gpt-oss-120B MaaS | global | medium reasoning, flat V13 judge contract |
| Judge B | Gemini 2.5 Pro | global | strict shallow response schema, thinking budget 128 |
| Adjudicator | Gemini 3.1 Pro Preview | global | blind disagreement resolution |

All model inference is remote through Vertex AI. The Colab Enterprise runtime acts as the controller.
