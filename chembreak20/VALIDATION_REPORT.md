# ChemBreak20 v20.0.0 validation report

Validation scope: offline/unit/integration validation of the packaged code. No live OpenAI, Vertex AI, ChemDFM, or ChemLLM production experiment was executed during packaging.

## Locked protocol

- Namespace: `CB20`
- Experiment revision: `CB20_ROUTE_MDP_MINI24_V1`
- Fixed panel: 24 non-reserve source tasks
- Learning: 3 epochs, at most 4 target turns per task episode
- Epsilon schedule: `0.30 -> 0.20 -> 0.15`
- Terminal evaluation: frozen controller, at most 4 target attempts
- Attack LLM: `gpt-5.6-sol`
- Judge LLM: `gemini-3.8-flash`
- Targets: ChemDFM and ChemLLM with fully independent controller/checkpoint state
- ChemDFM dialogue format: documented round-based Human/Assistant template
- ChemLLM dialogue format: tokenizer-native InternLM2 chat template, with explicit model-card-compatible fallback

## Checks completed

- Unit/integration suite: **41 passed**
- Package validator: **OK**
- Exactly one notebook under `notebooks/`
- Notebook code cells parse successfully and contain no saved outputs/execution counts
- Fixed 24-task panel reproduced exactly from the bundled 500-task source bank
- Taxonomy coverage: **9 HC / 8 HD / 15 OT**, zero reserve rows
- No embedded OpenAI API key detected
- No old-version runtime storage/config/checkpoint namespace references detected
- Attack-LLM policy blocks do not consume target turns and do not update Q/route memory
- General Attack-LLM and candidate-judge errors are isolated at the action level
- Target-generation errors are logged and do not crash the entire target experiment
- Learning-episode crash rollback restores pre-episode Q/route state before resume
- Terminal stage uses two distinct learned-route attempts before synthesized frozen-policy fallbacks when routes are available
- Terminal stage performs no Q/route updates
- ChemDFM and ChemLLM target-state directories are independent

## Full dry-run integration

Using deterministic mock roles and mock targets with all 24 tasks:

| Target | Episodes | Target turns | Q updates | Route updates |
|---|---:|---:|---:|---:|
| ChemDFM | 120 | 408 | 288 | 72 |
| ChemLLM | 120 | 408 | 288 | 72 |
| **Total** | **240** | **816** | **576** | **144** |

A second run against the same checkpoints remained at the same counts for both targets, confirming that completed work was not duplicated on resume.

## Live-runtime boundary

The package still requires the real cloud runtime to validate model download/authentication, GPU memory compatibility, live OpenAI/Vertex responses, provider policy behavior, and the exact target-model outputs. The notebook runs a live preflight before production execution, including the external roles and both target tokenizers.
