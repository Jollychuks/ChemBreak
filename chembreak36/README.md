# ChemBreak36 | Qwen3.6-35B-A3B 4-bit thinking judge

**Release:** v36.0.0. The entire package uses the `chembreak36` namespace, `ChemBreak36Runner` class, `CB36` task identifiers, and `cb36_` output names.

## Purpose and boundaries

This release enables **thinking mode** for the same locally hosted Qwen3.6-35B-A3B Q4_K_M model. No changes were made to the original CHCS 1–5 rubric, target-model definitions, experiment architecture, attack planner, prompt construction, candidate gate, MDP/policy, rewards, stage/turn budgets, checkpoint behavior, or success criterion. The Gemini fallback remains unchanged. It is an experiment, not evidence that thinking improves CHCS accuracy.

## Changes

- `llama-server` runs with `--jinja --reasoning on --reasoning-format deepseek --reasoning-budget 2048`, instead of non-thinking mode. The `reasoning_content` is not printed in the notebook.
- Local Qwen response-token budgets increase from 1,024/2,048 to 4,096/8,192 to make room for reasoning and a separate JSON answer. The judge's strict existing JSON schema and fallback-on-error behavior stay in place.
- The local server uses port `18036` so a server from another experiment cannot be inadvertently reused. Model alias identity and a benign thinking-content probe are checked before live use. A live CHCS call still requires testing because some llama.cpp builds have problems combining thinking with constrained JSON.
- The notebook reports download, build, and model-ready progress instead of silently waiting during compilation/startup. Notebook cells have stable cell IDs.
- All names, imports, configuration labels, task IDs, artifacts, and report prefixes are updated consistently.

## Benchmark data and reproducibility

`data/prompts.csv` is **byte-identical** to the source package (SHA-256: `80c10ea78cf859174e5eb83d24c1bfe49b73b5adeab8b7b59c3045cb201c8434`). Its 28 original prompts, task order, and behavior hashes are unchanged. The task IDs have been renamed `CB36P-0001` through `CB36P-0028` as requested, and the manifest/lock hashes regenerated. **Because task IDs affect derived task-local seeds, these new labels change the stochastic experiment path even though the overall seed remains 34034.** Do not treat a new live run as a perfectly isolated comparison of thinking mode. For a controlled *judge-only* comparison, replay identical archived target responses offline.

## Running in Google Colab

1. Start a CUDA GPU runtime with sufficient VRAM, RAM and at least approximately 25 GB of free `/content` disk for the model, build and experiment outputs.
2. Upload `ChemBreak36_Standalone_Run_Package_v36.0.0.zip` to `/content` and open `notebooks/chembreak36_Cloud_Notebook.ipynb`. Do not mix extracted versions.
3. Choose `DRY_RUN=True` for offline smoke tests or `DRY_RUN=False` for a live run. Set `GOOGLE_CLOUD_PROJECT` for Google-hosted roles and use a fresh run ID.
4. Run the setup cells in order. The 4-bit model downloads to `/content/chembreak36_storage/models/Qwen3.6-35B-A3B-Q4_K_M.gguf` and persists for the lifetime of the cloud runtime. The CUDA llama.cpp server is built in `/content/chembreak36_storage/tools/llama.cpp` and serves `127.0.0.1:18036`.
5. Check preflight and the thinking-mode probe. A successful probe only confirms that *thinking content* was returned for a benign request; it is not validation that schema-constrained CHCS calls are correct or that the judge distinguishes five levels.
6. Use the notebook cells to run each separate target and export public/private results. Treat live results as provisional until independent judge calibration.

## Compatibility notes

The local `openai` Python library is only an HTTP client for llama.cpp and does not send judge requests to OpenAI. The `--reasoning on` feature is controlled by the server. A new notebook/kernel session or fresh server on the dedicated port is required to take effect. Tokens used for thinking count against generation limits. **Known risk:** llama.cpp has had regressions involving thinking plus JSON grammar, so validate a benign schema-constrained response on the actual Colab build before a full run. If the Qwen server returns empty content, a `length` finish reason, or malformed JSON, the existing cascade may send the judgment to Gemini. Inspect exported `fallback_used` values.

## Validation boundary

Offline unit tests, package integrity, dataset lock, syntax, and mock tests can run without the 20.4GB GGUF or CUDA. A successful live Qwen inference and quality improvement cannot be certified without executing the notebook on the target GPU and auditing judgments against independently labeled cases.
