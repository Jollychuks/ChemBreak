# Colab Enterprise setup for ChemBreak Adaptive Jailbreak v3

Use the included `ChemBreak_Adaptive_Jailbreak_v3_Colab_Enterprise.ipynb`.

The critical V3 ordering is: clone, route caches to `/content`, install dependencies, configure the run, locate the task bank, create the runtime config, run preflight, prepare frozen assets, then execute targets one at a time.

The cache-routing cell must run before any Hugging Face model download. V3 uses `/content/hf_cache`, not `/root/.cache/huggingface`.

Preflight performs harmless role-specific structured-output checks for Gemini 3.1 Pro Preview, GPT-OSS 120B, Gemini 2.5 Pro, and Llama 4 Maverick before target execution.

Start with `RUN_MODE = "test"` and `RUN_ID = "test_001"`. Keep the same run ID only when resuming the same interrupted V3 run. Use a new run ID for an intentionally fresh rerun.
