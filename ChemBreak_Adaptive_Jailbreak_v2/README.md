# ChemBreak Adaptive Jailbreak v2

GitHub-to-Colab Enterprise package for the controlled and adaptive ChemBreak chemistry-LLM safety experiment.

Repository folder: `ChemBreak_Adaptive_Jailbreak_v2/`

## Why V2 exists

The v1.1.1 TEST run successfully prepared all 8 frozen attack assets and loaded ChemDFM, but exposed three infrastructure problems: the default Hugging Face cache filled the small system disk while `/content` remained largely unused, judge calls sometimes returned non-parseable JSON or were throttled, and a target-load failure was repeated across the progress display. V2 fixes those infrastructure problems while preserving the experimental methodology.

## Experimental design

Controlled evaluation:

- C0 Direct Single-Turn
- C1 Repeated Single-Turn, fresh target conversation per attempt
- C2 Fixed Multi-Turn, full sequence frozen before target feedback

Adaptive evaluation:

- C3 Adaptive ChemBreak, response-adaptive with CONTINUE, REFINE, SWITCH, or STOP

The four adaptive route families remain Representation, Context, Conceptual Decomposition, and Output-Oriented. Judge feedback is not sent to the attacker in the primary experiment.

## Model roles

| Role | Model |
|---|---|
| Attacker and attack-graph builder | `gemini-3.1-pro-preview` |
| Safety / goal judge | `openai/gpt-oss-120b-maas` |
| Chemistry judge | `gemini-2.5-pro` |
| Chemistry Domain Verifier | deterministic RDKit + task metadata |
| Adjudicator | `meta/llama-4-maverick-17b-128e-instruct-maas` |
| Targets | ChemDFM, ChemLLM, LlaSMol |

## Sampling

- TEST: 8 unique task-bank prompts
- PILOT: 40 unique task-bank prompts
- PRODUCTION: all 500 final task-bank prompts

Selection is stratified by `hc_category` and `hazard_domain` where available.

## Storage fix

The target model cache is explicitly located at:

```text
/content/hf_cache
```

V2 passes that path directly to target tokenizer/model/adapter loaders and also configures the common Hugging Face, Transformers, Xet, Torch, and temporary-directory environment variables. Preflight blocks execution if the cache filesystem has less than 80 GB free or the system disk has less than 5 GB free.

The notebook can remove the obsolete `/root/.cache/huggingface` directory on the ephemeral runtime when it is not the configured V2 cache. It never deletes GCS data or modifies GitHub.

## Judge and quota reliability

Cloud model clients now separate network retries from JSON/schema retries. GPT-OSS and Maverick use bounded exponential backoff for HTTP 429 and other retryable server responses. Judge outputs are retried when required score keys are missing, scores fall outside 0 to 5, JSON is malformed, or adjudicator booleans are missing.

These retries are infrastructure handling only. They do not change the attack budget because judge calls are not target-model queries.

## Target loading

Each local target is loaded once per target block and reused across all selected tasks and conditions. If loading fails, V2 records one `target_load` error, checkpoints it, and aborts that target cell immediately. A model-load failure is treated as infrastructure failure, not as a benchmark refusal or success.

## Chemistry Domain Verifier

The verifier is not another LLM. It conservatively uses task metadata plus RDKit checks for explicitly represented chemistry. It can return `VERIFIED`, `PARTIALLY_VERIFIED`, `UNVERIFIABLE`, or `CONTRADICTED`. `UNVERIFIABLE` is not automatically a failure.

## Google Cloud workflow

1. Upload this entire folder to the root of `Jollychuks/ChemBreak` and commit it yourself.
2. Open `ChemBreak_Adaptive_Jailbreak_v2_Colab_Enterprise.ipynb` in Colab Enterprise.
3. Connect the A100 runtime.
4. Run the notebook from the top, including the storage-routing cell before dependency installation.
5. Start with `RUN_MODE = "test"` and `RUN_ID = "test_001"`.
6. Run PILOT only after the TEST is clean. Freeze methodology after PILOT, then run PRODUCTION.

## Output namespace

Local:

```text
outputs/CB-ADAPTIVE-JAILBREAK-V2/<mode>/<run_id>/
```

GCS:

```text
gs://<bucket>/ChemBreak_Adaptive_Jailbreak_v2/<mode>/<run_id>/
```

The V2 namespace does not overwrite v1, v1.1, or v1.1.1.

## Public vs restricted output

Keep the completed harmful task bank and raw target transcripts out of the public GitHub repository. Restricted data remains under the configured GCS output prefix. Public summaries and manifests are written separately.

## Commands

After `configs/runtime.yaml` is created:

```bash
python scripts/preflight.py --config configs/runtime.yaml
python scripts/run.py prepare --config configs/runtime.yaml
python scripts/run.py execute --config configs/runtime.yaml --target chemdfm
python scripts/run.py execute --config configs/runtime.yaml --target chemllm
python scripts/run.py execute --config configs/runtime.yaml --target llasmol
python scripts/run.py metrics --config configs/runtime.yaml
```

See `VERSION_2_NOTES.md` for the reliability changes and `AUDIT.md` for offline validation.
