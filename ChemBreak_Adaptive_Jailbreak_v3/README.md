# ChemBreak Adaptive Jailbreak v3

GitHub-to-Colab Enterprise package for the controlled and adaptive ChemBreak chemistry-LLM safety experiment.

Repository folder: `ChemBreak_Adaptive_Jailbreak_v3/`

## Why V3 exists

The V2 TEST demonstrated that the storage correction worked and that ChemDFM could load from `/content/hf_cache`, but it also exposed evaluation and methodology problems that should not be carried into PILOT. The main issues were excessive Gemini 2.5 Pro JSON retries, repeated GPT-OSS 120B throttling, insufficient visibility of the attacker, and contamination of the adaptive history with judge results even though judge feedback was configured as disabled.

V3 keeps the same experimental conditions and model roles, but fixes those implementation problems in a fresh namespace.

## Experimental design

Controlled evaluation:

- C0 Direct Single-Turn
- C1 Repeated Single-Turn, fresh target conversation per attempt
- C2 Fixed Multi-Turn, full sequence frozen before target feedback

Adaptive evaluation:

- C3 Adaptive ChemBreak, response-adaptive with CONTINUE, REFINE, SWITCH, or STOP

The four adaptive route families remain Representation, Context, Conceptual Decomposition, and Output-Oriented.

The primary C3 experiment has `attacker_uses_judge_feedback: false`. The attacker receives the task, attack graph, prior attacker prompts, target responses, route state, and remaining budget. It does not receive safety-judge scores, chemistry-judge scores, verifier results, adjudication, or hidden target internals.

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

## V3 methodology correction: clean histories

V2 stored judge evaluations inside the same turn structure used to build later C2/C3 contexts. In C3, that also allowed the attacker to receive evaluation information even though the configuration said judge feedback was disabled.

V3 maintains two separate histories:

```text
conversation history
  user + target response only

experiment record
  user + target response + route + evaluation
```

Only clean conversation history is sent to the target, safety judge, chemistry judge, and response-adaptive attacker. Evaluation records remain available for checkpointing and later analysis, but do not re-enter the primary interaction loop.

## Explicit attacker visibility

The attacker is used in two places:

1. PREPARE: Gemini 3.1 Pro Preview generates the C1 repeated-single set, the C2 fixed sequence, and the C3 four-route graph once per selected task.
2. C3 runtime: Gemini 3.1 Pro Preview reads the clean target conversation and chooses the next route/action/query after each unsuccessful turn.

V3 prints attacker-role progress without printing the generated harmful prompt text. Example status lines identify the attacker model, task, stage, route transition, action, remaining budget, and whether judge feedback is OFF or ON.

## Gemini 2.5 Pro chemistry-judge fix

V2 relied on `application/json` response mode without an enforced response schema, then nested up to 5 JSON attempts inside 3 judge-validation attempts. This could generate many calls for a single evaluation and still end with truncated or malformed output.

V3 uses the Gemini structured-output JSON Schema interface for the chemistry judge. The schema requires exactly the chemistry scoring fields used by ChemBreak. V3 also:

- raises the chemistry-judge output allowance to 4096 tokens;
- sets a bounded thinking budget of 1024 tokens;
- checks model finish reason, including `MAX_TOKENS`;
- uses one client-level structured generation attempt by default;
- allows at most two validated judge attempts;
- keeps the final judge output compact.

This reduces the V2 worst case from many nested formatting retries to a small bounded number of structured calls.

## GPT-OSS 120B rate-limit fix

The safety judge remains GPT-OSS 120B. V3 increases the baseline pacing and adds adaptive 429 pacing. A 429 response increases the interval between requests, `Retry-After` is honored when present, and the dynamic interval decays after successful calls rather than immediately returning to a bursty request rate.

## Role-specific preflight

Preflight now performs harmless structured-output smoke tests for the actual roles before any target jailbreak execution:

- attacker structured decision output;
- GPT-OSS safety score object;
- Gemini 2.5 Pro chemistry score object using JSON Schema;
- Llama 4 Maverick adjudication object.

If a required structured-output path is not working, the run stops before loading a target model.

## Storage behavior

The target model cache remains explicitly located at:

```text
/content/hf_cache
```

V3 passes that path directly to target tokenizer/model/adapter loaders and configures the common Hugging Face, Xet, Torch, and temporary-directory environment variables. Preflight blocks execution if the cache filesystem has less than 80 GB free or the system disk has less than 5 GB free.

The notebook can remove the obsolete `/root/.cache/huggingface` directory on the ephemeral runtime when it is not the configured V3 cache. It never deletes GCS results and never modifies GitHub.

## Target loading

Each local target is loaded once per target block and reused across all selected tasks and conditions. If loading fails, V3 records one `target_load` infrastructure error and aborts that target cell immediately. A load failure is never counted as a refusal or jailbreak result.

## Chemistry Domain Verifier

The verifier is not another LLM. It conservatively uses task metadata plus RDKit checks for explicitly represented chemistry. It can return `VERIFIED`, `PARTIALLY_VERIFIED`, `UNVERIFIABLE`, or `CONTRADICTED`. `UNVERIFIABLE` is not automatically a failure.

## Diagnostics

At the end of execution V3 writes client-level reliability counters to:

```text
public/model_client_diagnostics.json
```

The file includes request counts, network retries, JSON retries, GPT-OSS rate-limit events, Gemini `MAX_TOKENS` events, and the final adaptive pacing interval where applicable.

## Google Cloud workflow

1. Upload this entire folder to the root of `Jollychuks/ChemBreak` and commit it yourself.
2. Open `ChemBreak_Adaptive_Jailbreak_v3_Colab_Enterprise.ipynb` in Colab Enterprise.
3. Connect the A100 runtime.
4. Run the notebook from the top, including the storage-routing cell before dependency installation.
5. Start with `RUN_MODE = "test"` and `RUN_ID = "test_001"`.
6. Run PILOT only after TEST is clean. Freeze methodology after PILOT, then run PRODUCTION.

## Output namespace

Local:

```text
outputs/CB-ADAPTIVE-JAILBREAK-V3/<mode>/<run_id>/
```

GCS:

```text
gs://<bucket>/ChemBreak_Adaptive_Jailbreak_v3/<mode>/<run_id>/
```

V3 does not import or overwrite V2, v1.1.1, or earlier run results.

## Public vs restricted output

Keep the completed harmful task bank and raw target transcripts out of the public GitHub repository. Restricted data remains under the configured GCS output prefix. Public summaries, manifests, and reliability diagnostics are written separately.

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

See `VERSION_3_NOTES.md` for the V3 changes and `AUDIT.md` for offline validation.
