# ChemBreak Adaptive Jailbreak v2 release notes

V2 is a fresh major-version namespace created after the v1.1.1 TEST run exposed infrastructure reliability issues. It does not overwrite earlier package folders or GCS outputs.

## V2 fixes

1. **Hugging Face cache moved to `/content/hf_cache`.** All target tokenizer/model/adapter loaders receive the cache path explicitly. The notebook also sets `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`, `HF_XET_CACHE`, `TORCH_HOME`, and `TMPDIR` before any model downloads.
2. **Disk preflight.** V2 prints system, `/content`, and HF-cache disk usage and blocks execution when configured free-space thresholds are not met.
3. **Legacy ephemeral cache cleanup.** The notebook/preflight can remove only the old `/root/.cache/huggingface` directory when it is not the configured cache. It never deletes GCS data or GitHub content.
4. **GPT-OSS throttling retry.** HTTP 429, 408, 409, 500, 502, 503, and 504 responses use bounded exponential backoff with jitter and `Retry-After` support.
5. **Judge JSON robustness.** Cloud clients have parse retries, and the evaluator adds schema-level validation retries for required judge scores and adjudicator booleans.
6. **Model pacing.** GPT-OSS and Maverick calls are paced by default to reduce immediate quota/throttling collisions.
7. **Target-load fail fast.** A local target that cannot load is attempted once per target block. The cell aborts with one clear error and the failure is checkpointed, rather than printing the same load failure for every experimental unit.
8. **Fresh namespace.** Folder, package version, output path, manifest, notebook, and run signature are V2-specific.

## Unchanged methodology

The controlled conditions C0, C1, C2, adaptive C3 logic, four route families, three target models, attack budgets, model-role assignments, chemistry domain verifier, success thresholds, and TEST/PILOT/PRODUCTION sampling remain unchanged from the frozen design.
