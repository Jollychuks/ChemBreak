# ChemBreak V9 Cloud

GitHub-connected ChemBreak V9 for Google Colab + Vertex AI.

Place the entire folder at:

`ChemBreak/ChemBreak_V9_Cloud/`

The Colab notebook clones the repository and uses only this folder.

## V9 structure

- `ChemBreak_V9_Google_Colab.ipynb`
- `scripts/chembreak_v9_cloud.py`
- `config/run_config.json`
- `prompts/`
- `taxonomy/`
- `data/`
- `outputs/`
- `requirements.txt`

## Default model roles

Generation:
- Gemini 3.1 Pro Preview
- Llama 4 Maverick
- gpt-oss-120B

Repair:
- gpt-oss-120B

Judging:
- Gemini 2.5 Pro
- gpt-oss-120B

Adjudication:
- Gemini 3.1 Pro Preview

## Run order

preflight -> bootstrap -> plan -> generate -> validate -> repair -> judge -> refill -> judge -> finalize

No personal model API key is required.
