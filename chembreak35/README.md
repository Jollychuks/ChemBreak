# ChemBreak35 | Qwen3.6-35B-A3B Q4_K_M local CHCS judge

**Package:** ChemBreak35 v35.0.1 (clean namespace rebuild)

This release uses the **ChemBreak35** Python package, `ChemBreak35Runner`, `CB35` experiment namespace, `CB35P-0001` through `CB35P-0028` task IDs, and `cb35_*` export filenames throughout. No previous-version names or import paths are required.

## Scope of changes

The judge substitution uses **Qwen3.6-35B-A3B Q4_K_M GGUF**, served locally with llama.cpp. The fallback judge is Gemini 3.8 Flash. Planner, intent gate, target configurations, stage counts, permitted actions, CHCS rubric and range, stopping conditions, reward logic, and evaluation workflow are unchanged. An inactive API-based judge class has been removed because it is not used by the local-Qwen configuration.

**Immutable task content:** `data/prompts.csv` is byte-identical to the supplied package (SHA-256 `80c10ea78cf859174e5eb83d24c1bfe49b73b5adeab8b7b59c3045cb201c8434`). The manifest and lock have been regenerated solely to replace the experiment namespace and task IDs. Behavior hashes, order, categories, and 28 source tasks are unchanged. New task IDs cause the per-task derived seeds to differ from previous experiment runs even though the global seed remains 34034. Use task order and behavior hash to match historical results, not task ID. **Do not reuse an existing checkpoint.**

## Local model storage

The 4-bit GGUF model is **not** included in the ZIP. The notebook downloads the approximately 20.4 GB file from:

https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF/blob/main/Qwen3.6-35B-A3B-Q4_K_M.gguf

The notebook saves it under `/content/chembreak35_storage/models/Qwen3.6-35B-A3B-Q4_K_M.gguf` and builds the CUDA llama.cpp server under `/content/chembreak35_storage/tools/llama.cpp`. The local server listens on `127.0.0.1:18035`, explicitly advertises the configured Qwen model alias, and provides the primary CHCS judgment. The notebook and preflight check the actual advertised model identity rather than accepting any server at that port. Non-thinking mode is set with the current `--reasoning off` llama.cpp flag, without the deprecated template-kwargs option. `openai` in Python dependencies is an HTTP client for the **local** OpenAI-compatible API and does not mean that the primary judge uses an OpenAI model or sends prompts to OpenAI.

## Cloud notebook run instructions

1. Start a CUDA-enabled cloud notebook instance, ideally A100 with adequate system RAM and free `/content` disk space.
2. Upload `ChemBreak35_Standalone_Run_Package_v35.0.1_clean.zip` to `/content`, or extract its `chembreak35/` directory there. Use `notebooks/chembreak35_Cloud_Notebook.ipynb`.
3. Select `DRY_RUN=False` for a live run. Set your `GOOGLE_CLOUD_PROJECT` in the environment for the Gemini planner, gate, and fallback. Use a **new** `RUN_ID`. The existing `SEED=34034` is retained as the global experiment seed.
4. Run the installation, configuration, model-download, and llama.cpp server startup cells. The GGUF download occurs directly on the cloud VM's `/content` filesystem and is reused if the file is already present.
5. Run preflight; optionally enable `PROBE_ROLES` and `PROBE_TOKENIZERS`. Run ChemDFM and ChemLLM in their separate cells; use the export cells to produce public and private result ZIPs.

**GPU memory:** model files on disk are separate from available GPU VRAM. The notebook defaults to 8 Qwen layers on GPU to help it coexist with the unmodified target models. That setting can make Qwen inference slower. For an A100 80GB you can increase it after checking free VRAM. Restart the local server to apply changed GPU layer settings.

**Validation boundary:** the code, notebook structure, lock/hash checks, and offline tests can be validated here. The 20.4 GB GGUF, CUDA build, Vertex credentials, and full end-to-end live judge inference have **not** been exercised in the user's Colab runtime. Conduct a benign live smoke test before a full experiment.

## Clean package inventory

- `configs/config.cb35.yaml`: experiment configuration
- `data/prompts.csv`: unchanged source benchmark tasks
- `data/CB35_prompts28_manifest_v1.csv` and `data/CB35_prompts28_lock_v1.json`: updated identifiers and matching hashes
- `src/chembreak35/`: canonical Python package and runner
- `notebooks/chembreak35_Cloud_Notebook.ipynb`: `/content` cloud run
- `tests/`: safe offline unit tests
- `requirements-cloud-ml.txt` and `pyproject.toml`: dependencies and install metadata

This is a version-label and task-identifier consistency refresh, not a redesign of the experiment or CHCS rubric.
