# ChemBreak 12 changes from ChemBreak 11

- Preserves Google Cloud Notebook Enterprise configuration with explicit PROJECT_ID / REPO_URL / BRANCH / PROJECT_SUBDIR.
- Uses fresh `/content/chembreak12_storage` paths; no CB7–CB11 runtime storage is imported.
- Replaces development/pilot/holdout/full_bank resampling with immutable Train/Test1/Test2/Test3/Test4/Reserve partitioning.
- Binds the run signature to the partition manifest hash.
- Verifies source-bank hash, prompt hashes, split counts, pairwise disjointness, complete coverage, and deterministic reproduction before live execution.
- Train uses a mutable policy artifact; Test1–Test4 require the frozen policy artifact.
- Keeps ChemDFM and ChemLLM as the active conversational target set.
