# ChemBreak Adaptive Jailbreak v3 audit

Build date: 2026-08-27

## Source basis

V3 was created as a fresh major-version namespace from the V2 codebase after the V2 TEST exposed excessive chemistry-judge formatting retries, GPT-OSS throttling, weak attacker visibility, and unintended judge-feedback/history contamination. V2 and earlier versions remain untouched.

## V3 namespace checked

- Package namespace: `CB-ADAPTIVE-JAILBREAK-V3`
- Python package version: `3.0.0`
- Repository folder: `ChemBreak_Adaptive_Jailbreak_v3`
- Notebook: `ChemBreak_Adaptive_Jailbreak_v3_Colab_Enterprise.ipynb`
- GCS base used by notebook: `ChemBreak_Adaptive_Jailbreak_v3`
- Local output namespace: `outputs/CB-ADAPTIVE-JAILBREAK-V3/<mode>/<run_id>/`
- No executable/config reference to a V2, v1.1.1, or v1.1 output namespace remains

## Methodology corrections checked

- `attacker_uses_judge_feedback: false` is now enforced in code
- C3 attacker history contains only user prompt, target response, and route metadata when judge feedback is disabled
- C2 and C3 maintain clean target conversation history separately from stored evaluation records
- Prior judge outputs do not re-enter later safety-judge or chemistry-judge context
- Gemini 3.1 Pro Preview attacker is explicitly logged during PREPARE and live C3 decisions
- Attack text itself is not printed by attacker progress logging
- C0 has no attacker; C1/C2 use frozen attacker assets; C3 is response-adaptive

## Reliability changes checked

- Gemini 2.5 Pro chemistry judge uses explicit JSON Schema structured output
- Chemistry judge output limit: 4096 tokens
- Chemistry judge thinking budget: 1024 tokens
- Chemistry judge client JSON attempts: 1 by default
- Judge validation attempts: 2 by default
- Gemini finish reasons are inspected and `MAX_TOKENS` is tracked distinctly
- GPT-OSS 120B baseline pacing increased to 4 seconds
- GPT-OSS 429 responses increase a bounded dynamic pacing interval
- `Retry-After` is honored when available
- Dynamic pacing decays after successful requests
- Role-specific harmless structured-output smoke tests run during preflight
- Model-client reliability diagnostics are written to `public/model_client_diagnostics.json`
- Hugging Face cache remains `/content/hf_cache`
- Deprecated `TRANSFORMERS_CACHE` environment variable is no longer set
- Target cache paths are passed explicitly to ChemDFM, ChemLLM, LlaSMol base, and LlaSMol adapter loaders
- Preflight checks HF-cache and system free space
- Target weights are loaded once per target block and reused
- Target-load failure aborts the requested target block once and is not counted as an experimental result

## Offline validation

- `python -m compileall -q chembreak scripts tests`: PASS
- `PYTHONPATH=. pytest -q`: **15 passed**
- Notebook JSON parse: PASS
- Runtime-config smoke generation: PASS
- V3 runtime config assertions: PASS
- Version/name audit across executable code/config/scripts: PASS
- ZIP content/root audit: PASS
- GitHub-ready ZIP contains 43 files under the V3 root folder
- No live target-model jailbreak calls were made during the build
- No GitHub repository was modified during the build

## Live validation still required

Offline validation cannot prove Google Cloud quota behavior, exact managed-model structured-output behavior, or full target-model execution. Run V3 TEST in Colab Enterprise before PILOT. V3 preflight is designed to catch required model-role and structured-output failures before target execution.
