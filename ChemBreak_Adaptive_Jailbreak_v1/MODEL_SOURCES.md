# Model source notes

Verified during package preparation on 2026-08-27.

- ChemDFM collection: https://huggingface.co/collections/OpenDFM/chemdfm
- Default ChemDFM: https://huggingface.co/OpenDFM/ChemDFM-v1.5-8B
- ChemLLM organization: https://huggingface.co/AI4Chem
- Default ChemLLM: https://huggingface.co/AI4Chem/ChemLLM-7B-Chat-1_5-SFT
- LlaSMol collection: https://huggingface.co/collections/osunlp/llasmol
- Default LlaSMol adapter: https://huggingface.co/osunlp/LlaSMol-Mistral-7B
- LlaSMol adapter config identifies its base as mistralai/Mistral-7B-v0.1.

The defaults prioritize a practical single-A100 run. Change model versions only before the pilot is frozen, then pin revisions for production.
