# GitHub upload checklist

Upload the entire folder `ChemBreak_Adaptive_Jailbreak_v1` to the repository root.

Expected top-level structure:

```text
ChemBreak/
├── ChemBreak_Colab_Generator_v1/
├── ChemBreak_Colab_Generator_v2/
├── ChemBreak_Colab_Generator_v3/
├── Data/
├── prompts/
└── ChemBreak_Adaptive_Jailbreak_v1/
    ├── ChemBreak_Adaptive_Jailbreak_v1_Colab_Enterprise.ipynb
    ├── chembreak/
    ├── configs/
    ├── data/
    ├── scripts/
    ├── tests/
    ├── requirements.txt
    ├── pyproject.toml
    └── README.md
```

Before committing, confirm that these are NOT included:

- `configs/runtime.yaml`
- `outputs/`
- `data/final_task_bank.csv`
- API keys, Hugging Face tokens, service-account JSON, or `.env` files

The final task bank and restricted transcripts should remain in private/restricted GCS.
