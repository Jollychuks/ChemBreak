# ChemBreak 12 architecture

ChemBreak 12 deliberately preserves the ChemBreak 11 Google Cloud Notebook Enterprise architecture.
The principal methodological change is the fixed pre-training partition manifest.

Flow:
1. Notebook user configuration: PROJECT_ID, REPO_URL, BRANCH, PROJECT_SUBDIR, PHASE, EXPERIMENT_REVISION, LIVE.
2. CB12-only storage/cache under `/content/chembreak12_storage`.
3. Clone/pull the GitHub repository and import `chembreak12` from `src/`.
4. Verify/reproduce the locked 500-task partition before any experiment starts.
5. Install the compatibility stack without replacing the image-provided Torch/CUDA stack.
6. Build a runtime config for exactly one phase: train, test1, test2, test3, or test4.
7. Run preflight: storage, imports, GPU, role schemas, tokenizers, partition lock.
8. Run C3_ADAPTIVE_MDP separately against ChemDFM and ChemLLM.
9. Freeze the learned policy after Train completes; Test1-Test4 use the frozen policy only.
10. Export release CSVs and a ZIP.

The repository intentionally does not bundle model weights, Hugging Face caches, CUDA caches, or the persistent runtime storage tree.
