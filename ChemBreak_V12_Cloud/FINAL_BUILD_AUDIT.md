# ChemBreak V12 Final Build Audit

Static audit: **PASS (85/85)**

## Mocked tests completed

- Vertex Gemini request payload carries `responseSchema`: PASS.
- Judge schema and normalization for A/B/C: PASS.
- Batch generation: two assignments produced six candidates using exactly two model calls: PASS.
- Concurrent two-judge mock: both judges produced four judgment rows across two assignments, two judge outcomes, and two selected tasks: PASS.
- Gemini Judge B mock required and received a strict response schema on every call: PASS.
- Prompt formatting smoke tests: PASS.

## Runtime change summary

- Normal initial generation calls fall from 3 per assignment in V11 to 1 per assignment in V12.
- Gemini Judge B now uses Vertex controlled JSON generation with an explicit response schema.
- Gemini generator, repair, refill, and adjudicator structured outputs also use response schemas where applicable.
- JSON-format fallback retries are reduced from two retries to one fallback retry.
- Two independent judges remain concurrent for each assignment.

## Limitation

This package was not executed against live Vertex AI endpoints during packaging. Run the fresh 9-task V12 test before pilot or production.

## Static checks

- **PASS**: Required file README.md
- **PASS**: Required file V12_DESIGN_LOCK.md
- **PASS**: Required file RELEASE_NOTES_V12.md
- **PASS**: Required file MODEL_ENDPOINTS.md
- **PASS**: Required file COLAB_ENTERPRISE_RUNBOOK.md
- **PASS**: Required file DEPLOYMENT_CHECKLIST.md
- **PASS**: Required file requirements.txt
- **PASS**: Required file ChemBreak_V12_Colab_Enterprise.ipynb
- **PASS**: Required file scripts/chembreak_v12_cloud.py
- **PASS**: Required file config/run_config.json
- **PASS**: Required file taxonomy/taxonomy_v12.json
- **PASS**: Required file prompts/generator_batch_template.txt
- **PASS**: Required file prompts/judge_system.txt
- **PASS**: Required file prompts/multi_candidate_judge_template.txt
- **PASS**: Required file prompts/single_candidate_judge_template.txt
- **PASS**: Required file ChemBreak_V12_Block_Diagram.png
- **PASS**: Required file ChemBreak_V12_Block_Diagram.svg
- **PASS**: No standard Google Colab notebook
- **PASS**: Pipeline compiles
- **PASS**: Pipeline AST parses
- **PASS**: Config version 12.0-cloud
- **PASS**: Pipeline version 12.0-cloud
- **PASS**: Fresh CBV12C namespace
- **PASS**: V12 assignment file
- **PASS**: V12 entity file
- **PASS**: V12 external reference file
- **PASS**: Test target 9
- **PASS**: Pilot 100 plus 15
- **PASS**: Production 500 plus 75
- **PASS**: HC1-HC9 preserved
- **PASS**: HD1-HD8 preserved
- **PASS**: OT1-OT15 preserved
- **PASS**: SC01-SC15 preserved
- **PASS**: Generator Gemini 3.1
- **PASS**: Repair Gemini 3.1
- **PASS**: Judge A gpt-oss-120B
- **PASS**: Judge B Gemini 2.5 Pro
- **PASS**: Adjudicator Gemini 3.1
- **PASS**: Exactly two judge roles
- **PASS**: No Llama model roles
- **PASS**: Vertex responseSchema supported in client
- **PASS**: Judge response schema helper
- **PASS**: Judge B receives strict schema
- **PASS**: Adjudicator receives strict schema
- **PASS**: Generator candidate schema helper
- **PASS**: Generator batch schema helper
- **PASS**: Initial generation batched by assignment
- **PASS**: Batch prompt loaded
- **PASS**: Batch generation enabled in config
- **PASS**: Structured generation enabled in config
- **PASS**: JSON fallback retry budget reduced
- **PASS**: Inter-call pacing reduced
- **PASS**: Judge B token budget compact
- **PASS**: Adjudicator token budget compact
- **PASS**: Transport retry retained
- **PASS**: 20s heartbeat retained
- **PASS**: Visual progress retained
- **PASS**: Concurrent two-judge code retained
- **PASS**: Two valid judgments required
- **PASS**: Technical pending retained
- **PASS**: Prejudge refill retained
- **PASS**: Single-candidate qualification retained
- **PASS**: Full refill retained
- **PASS**: Repair exact defects retained
- **PASS**: Refill full history retained
- **PASS**: Judge safer/sanitized warning retained
- **PASS**: 22-45 allowed, 30-40 preferred
- **PASS**: Near-duplicate threshold retained
- **PASS**: External similarity threshold retained
- **PASS**: Pipeline SHA in run signature
- **PASS**: Compatibility guard retained
- **PASS**: Source snapshot manifest retained
- **PASS**: Notebook valid JSON
- **PASS**: Notebook V12 project path
- **PASS**: Notebook uses existing project bucket
- **PASS**: Notebook does not create buckets
- **PASS**: Notebook does object access probe
- **PASS**: Notebook restores GCS checkpoints
- **PASS**: Notebook mirrors checkpoints every 60s
- **PASS**: Notebook streams child output
- **PASS**: Notebook test default
- **PASS**: Source manifest V12
- **PASS**: No prior generated tasks as generation input
- **PASS**: No V11 operational namespace leakage
- **PASS**: No em dash in release text
