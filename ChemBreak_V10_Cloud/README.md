# ChemBreak V10 Cloud

ChemBreak V10 is the GitHub-connected task-bank generator for Google Colab and Vertex AI.

Place the complete folder at:

`ChemBreak/ChemBreak_V10_Cloud/`

The notebook clones the repository and uses only this V10 folder.

## Folder structure

- `ChemBreak_V10_Google_Colab.ipynb`
- `ChemBreak_V10_Colab_Enterprise.ipynb`
- `scripts/chembreak_v10_cloud.py`
- `config/run_config.json`
- `prompts/`
- `taxonomy/`
- `data/`
- `outputs/`
- `MODEL_ENDPOINTS.md`
- `RELEASE_NOTES_V10.md`
- `requirements.txt`

## Model roles

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

## Run profiles

- Test: 9 final tasks
- Pilot: 100 final tasks plus 15 reserve assignments
- Production: 500 final tasks plus 75 reserve assignments

With three generators, production begins with up to 1,725 initial candidate calls before repair and refill.

## Run order

`preflight -> bootstrap -> plan -> inspect -> generate -> validate -> repair -> judge -> refill -> judge -> finalize`

Always complete test first, then pilot, then production.

## Persistence

The standard Google Colab notebook stores results under:

`MyDrive/ChemBreak_V10/outputs/<run_type>/`

Cloud Storage synchronization can also be enabled in the runtime configuration.

## Authentication

Standard Google Colab uses Application Default Credentials through `gcloud auth application-default login --no-launch-browser`.

No personal Gemini, Meta, OpenAI, or other model API key is required.

## Independence

V10 does not import or call any earlier ChemBreak code folder or generated task bank.


## Live progress

Long-running V10 stages continuously print progress to the notebook. Generation, repair, judging, adjudication, and refill show a heartbeat about every 20 seconds while waiting for Vertex AI. All major stages show completed counts, percentages, elapsed time, and ETA where meaningful.
