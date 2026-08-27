# GitHub upload checklist

Upload the entire folder `ChemBreak_Adaptive_Jailbreak_v1_1_1` to the repository root.

Expected structure:

```text
ChemBreak/
└── ChemBreak_Adaptive_Jailbreak_v1_1_1/
    ├── ChemBreak_Adaptive_Jailbreak_v1_1_1_Colab_Enterprise.ipynb
    ├── chembreak/
    ├── configs/
    ├── data/
    ├── scripts/
    ├── tests/
    ├── requirements.txt
    ├── pyproject.toml
    └── README.md
```

Do not commit:

- `configs/runtime.yaml`
- `outputs/`
- `data/final_task_bank.csv`
- API keys
- Hugging Face tokens
- service-account JSON
- `.env` files

The final task bank and restricted transcripts should remain in private or access-controlled GCS.
