# ChemBreak V13 Final Build Audit

Static and mocked audit: **PASS (87/87 checks)**

## What was tested

- V13 fresh namespace, output paths, and run signature controls.
- Model-role configuration and speed-oriented budgets.
- Flat Judge A / Judge B response contract.
- Compatibility with V12 nested and direct gpt-oss score shapes.
- Gemini 2.5 Pro shallow response schema and 128-token thinking budget configuration.
- Two independent concurrent judges and technical-failure non-vote behavior.
- Candidate-count recovery, repair, refill, and single-candidate qualification stages.
- Colab Enterprise notebook paths, existing bucket, `/content` working disk, and live streaming.
- Prompt-template formatting and Python compilation.

## Important limitation

This package was not executed against the user's live Vertex AI endpoints during packaging. The V13 preflight now includes an actual judge-contract smoke test so the 9-task test should not proceed unless both judge serialization contracts work live.

## Checks

- **PASS**: required file ChemBreak_V13_Colab_Enterprise.ipynb
- **PASS**: required file scripts/chembreak_v13_cloud.py
- **PASS**: required file config/run_config.json
- **PASS**: required file taxonomy/taxonomy_v13.json
- **PASS**: required file README.md
- **PASS**: required file V13_DESIGN_LOCK.md
- **PASS**: required file RELEASE_NOTES_V13.md
- **PASS**: required file MODEL_ENDPOINTS.md
- **PASS**: required file COLAB_ENTERPRISE_RUNBOOK.md
- **PASS**: required file DEPLOYMENT_CHECKLIST.md
- **PASS**: required file ChemBreak_V13_Block_Diagram.png
- **PASS**: required file ChemBreak_V13_Block_Diagram.svg
- **PASS**: required file prompts/judge_system.txt
- **PASS**: required file prompts/multi_candidate_judge_template.txt
- **PASS**: required file prompts/single_candidate_judge_template.txt
- **PASS**: version 13.0-cloud
- **PASS**: fresh CBV13C namespace
- **PASS**: test target 9
- **PASS**: pilot target/reserve 100/15
- **PASS**: production target/reserve 500/75
- **PASS**: three candidates per assignment
- **PASS**: batched initial generation retained
- **PASS**: generator Gemini 3.1
- **PASS**: repair Gemini 3.1
- **PASS**: Judge A gpt-oss
- **PASS**: Judge B Gemini 2.5 Pro
- **PASS**: adjudicator Gemini 3.1
- **PASS**: no Llama roles
- **PASS**: Judge A medium reasoning
- **PASS**: Judge A 1400 tokens
- **PASS**: Judge A no JSON model retry
- **PASS**: Judge B 1200 tokens
- **PASS**: Judge B min Pro thinking budget 128
- **PASS**: Judge B one JSON retry
- **PASS**: adjudicator 1200 tokens
- **PASS**: faster pacing
- **PASS**: transport retries 5
- **PASS**: 20 second heartbeat preserved
- **PASS**: two independent judges configured
- **PASS**: judge concurrency preserved
- **PASS**: two valid judgments required
- **PASS**: flat judge contract config
- **PASS**: contract preflight enabled
- **PASS**: thinkingConfig support in Vertex Gemini
- **PASS**: shallow judge schema
- **PASS**: robust score array normalizer
- **PASS**: V11/V12 nested/direct compatibility
- **PASS**: technical judge failure remains non-vote
- **PASS**: local contract failure does not force model retry
- **PASS**: judge contract preflight uses both judges
- **PASS**: prompt bounds 22-45 preferred 30-40
- **PASS**: near duplicate threshold
- **PASS**: external reference threshold
- **PASS**: deterministic entity validation
- **PASS**: taxonomy leak detection
- **PASS**: answer leakage detection
- **PASS**: mixed deliverables detection
- **PASS**: contradiction detection
- **PASS**: repair attempts 2
- **PASS**: prejudge refill attempts 2
- **PASS**: full refill 2 candidates
- **PASS**: full refill max cycles 3
- **PASS**: prejudge refill stage exists
- **PASS**: single candidate qualification exists
- **PASS**: full refill stage exists
- **PASS**: run signature contains pipeline hash
- **PASS**: compatibility guard exists
- **PASS**: source snapshot manifest exists
- **PASS**: V13 versioned assignment/entity/ref files
- **PASS**: HC1-HC9 preserved
- **PASS**: HD1-HD8 preserved
- **PASS**: OT1-OT15 preserved
- **PASS**: SC01-SC15 preserved
- **PASS**: notebook JSON valid
- **PASS**: notebook V13 project path
- **PASS**: notebook existing project bucket
- **PASS**: notebook V13 GCS prefix
- **PASS**: notebook no bucket creation
- **PASS**: notebook uses /content working disk
- **PASS**: notebook line streaming
- **PASS**: notebook all stages
- **PASS**: notebook verifies V13 score normalizer
- **PASS**: all prompt templates format
- **PASS**: mock judge compatibility tests
- **PASS**: no operational V12 namespace leakage
- **PASS**: no em dash in release text
- **PASS**: pipeline compiles
