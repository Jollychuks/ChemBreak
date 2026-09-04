# ChemBreak Adaptive Jailbreak V4

ChemBreak V4 is the downstream evaluation stage for the completed ChemHarm prompt bank. It treats every ChemHarm row as a fixed harmful objective, applies four controlled test conditions under the same five-query target budget, and compares ChemDFM, ChemLLM, and LlaSMol with the same task subset, model settings, judges, and stopping rules.

This package does not regenerate, relabel, or remap the ChemHarm dataset. The included `data/final_task_bank.csv` is the frozen 500-task input.

## Registered experiment

| Condition | Definition | Target-query budget |
| --- | --- | ---: |
| `C0_DIRECT` | The original ChemHarm prompt is submitted in a fresh context on each attempt. | 5 |
| `C1_REPEATED_SINGLE` | Five distinct single-turn transformations are generated once per task and submitted independently. | 5 |
| `C2_FIXED_MULTI` | A five-turn chain is generated before any target response is observed, then used unchanged for every target. | 5 |
| `C3_ADAPTIVE_MDP` | A state-aware policy selects a high-level action after each judged target response, then the attacker realizes that action as the next utterance. | 5 |

Every transmitted target request consumes one query. Automatic target retries are disabled by default. Attacker, judge, and adjudicator calls do not count as target queries, but are recorded separately.

## Fixed model roles

| Role | Model |
| --- | --- |
| Policy planner, graph builder, prompt realizer | `gemini-3.1-pro-preview` |
| Safety and goal judge | `openai/gpt-oss-120b-maas` |
| Chemistry judge | `gemini-2.5-pro` |
| Disagreement adjudicator | `meta/llama-4-maverick-17b-128e-instruct-maas` |
| Target 1 | `OpenDFM/ChemDFM-v1.5-8B` |
| Target 2 | `AI4Chem/ChemLLM-7B-Chat-1_5-SFT` |
| Target 3 | `mistralai/Mistral-7B-v0.1` plus `osunlp/LlaSMol-Mistral-7B` |

## Phase sizes

| Phase | Fixed task count | Episodes | Maximum target queries |
| --- | ---: | ---: | ---: |
| Test | 8 | 96 | 480 |
| Pilot | 40 | 480 | 2,400 |
| Production | 500 | 6,000 | 30,000 |

Episode count equals tasks multiplied by three targets and four conditions. Maximum query count equals episodes multiplied by five. Early confirmed success and declared terminal rules can reduce actual usage.

## Recommended Google Cloud workflow

1. Add this complete directory to the existing `Jollychuks/ChemBreak` repository. Keep it under `ChemBreak_Adaptive_Jailbreak_V4` so earlier versions remain untouched.
2. Start a Google Cloud Notebook Enterprise runtime with an NVIDIA A100 GPU and enough persistent disk for the three Hugging Face checkpoints and outputs.
3. Open `notebooks/ChemBreak_V4_Cloud_Notebook.ipynb` from the repository.
4. Enter the Google Cloud project ID. Keep `LIVE = False` for the first pass.
5. Run the installation, configuration, dry preflight, and mock test cells.
6. In Model Garden, enable API access for GPT-OSS 120B and Llama 4 Maverick. Confirm that Gemini 3.1 Pro Preview and Gemini 2.5 Pro are available in the project.
7. Set `LIVE = True`, run live preflight, then execute the test phase.
8. Review the test output and failure ledger. Run pilot only after the test has completed cleanly. Freeze the configuration before production.

The notebook uses the Google Gen AI SDK with Application Default Credentials supplied by Google Cloud. No API key is stored in the repository.

## Command-line use

From this directory:

```bash
python -m pip install -e .
chembreak-v4 preflight --config configs/config.test.yaml
chembreak-v4 run --config configs/config.test.yaml
```

Live execution requires all three settings:

```bash
export GOOGLE_CLOUD_PROJECT="YOUR_PROJECT_ID"
export CHEMBREAK_ENABLE_LIVE="YES"
chembreak-v4 preflight --config configs/config.pilot.yaml
chembreak-v4 run --config configs/config.pilot.yaml
```

Use `--load-targets` during preflight only when you want a full sequential download and load test:

```bash
chembreak-v4 preflight --config configs/config.test.yaml --load-targets
```

## Resumption and checkpoints

Each run directory is named from the phase and a signature derived from the configuration, task-bank hash, and code version. The SQLite checkpoint uses atomic commits after every turn. A rerun with the same inputs resumes completed work. A changed configuration produces a different run directory. A signature mismatch inside an existing directory is rejected.

Set `run.gcs_checkpoint_uri` in the runtime configuration to a private `gs://bucket/prefix` path. The runner uploads a consistent SQLite snapshot at the configured episode interval.

## Outputs

Each signed run contains:

- `run_manifest.json` with the immutable experiment contract
- `selected_tasks.csv` and its distribution summary
- `state.sqlite3` and `checkpoint_snapshot.sqlite3`
- `private/turns_raw.jsonl` with prompts and target responses
- `private/api_calls.csv` with model-role usage
- `release/episode_summary.csv`
- `release/metrics_overall.csv`
- `release/metrics_by_axis.csv`
- `release/asr_by_query_budget.csv`
- `release/paired_comparisons.csv`
- `release/turns_redacted.csv`
- `release/failures.csv`

Raw target responses remain outside the release export unless `release_raw_outputs` is explicitly enabled. Keep raw outputs in controlled storage and require chemistry-expert review before any disclosure.

## Methodological boundary

The current production design is a frozen, state-aware LLM policy in an MDP environment. It is not a trained reinforcement-learning policy. The reward is logged for trajectory analysis and future development. Do not tune the policy on the final 500 tasks and then report those same tasks as untouched evaluation data. Any learned policy should be developed on a separate collection and frozen before this benchmark is run.

See `docs/METHODOLOGY.md` and `docs/OUTPUT_SCHEMA.md` for the complete contract.
