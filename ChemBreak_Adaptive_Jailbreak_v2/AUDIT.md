# ChemBreak Adaptive Jailbreak v2 audit

Build date: 2026-08-27

## Source basis

V2 was rebuilt from the last GitHub-ready v1.1.1 package, with a fresh major-version namespace. The experimental design and role assignments were retained while infrastructure reliability was revised in response to the v1.1.1 cloud TEST run.

## Reliability changes checked

- V2 namespace: `CB-ADAPTIVE-JAILBREAK-V2`
- Python package version: `2.0.0`
- Repository folder: `ChemBreak_Adaptive_Jailbreak_v2`
- Notebook: `ChemBreak_Adaptive_Jailbreak_v2_Colab_Enterprise.ipynb`
- GCS base used by notebook: `ChemBreak_Adaptive_Jailbreak_v2`
- Hugging Face cache: `/content/hf_cache`
- Explicit target `cache_dir` wiring for ChemDFM, ChemLLM, LlaSMol base, and LlaSMol adapter
- Early cache environment routing before target-model imports
- Optional cleanup of obsolete `/root/.cache/huggingface` on the ephemeral runtime
- Preflight free-space checks for HF-cache and system filesystems
- Retryable HTTP status handling including 429 with exponential backoff and jitter
- JSON parsing retries for cloud-model clients
- Schema-level retry for safety judge, chemistry judge, and adjudicator output
- One-line target-block abort and nonzero cell failure when a requested target cannot load
- Attack-budget methodology unchanged

## Offline validation

- `python -m compileall -q chembreak scripts tests`: PASS
- `PYTHONPATH=. pytest -q`: **12 passed**
- Notebook JSON parse: PASS
- Runtime-config smoke generation: PASS
- Version/name audit across executable code/config/scripts: PASS
- No live target-model jailbreak calls were made during the build
- No GitHub repository was modified during the build

## Live validation still required

Offline validation cannot prove Google Cloud quota behavior, exact managed-model response behavior, or full target weight loading. Run V2 TEST in Colab Enterprise before PILOT. The preflight is designed to fail before target execution if storage or required managed-model connectivity is not ready.
