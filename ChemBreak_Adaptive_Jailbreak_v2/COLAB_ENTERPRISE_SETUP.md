# Colab Enterprise setup for ChemBreak Adaptive Jailbreak v2

Use the included `ChemBreak_Adaptive_Jailbreak_v2_Colab_Enterprise.ipynb`.

The critical V2 ordering is: clone, route caches to `/content`, install dependencies, configure run, locate task bank, create runtime config, preflight, prepare assets, then run targets one at a time.

The cache-routing cell must run before any Hugging Face model download. V2 uses `/content/hf_cache`, not the default `/root/.cache/huggingface` path that filled the smaller system disk in the previous test.

Start with `RUN_MODE = "test"` and `RUN_ID = "test_001"`. Keep the same run ID only when resuming the same interrupted V2 run.
