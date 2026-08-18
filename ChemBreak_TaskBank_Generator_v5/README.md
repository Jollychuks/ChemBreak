# ChemBreak Task Bank Generator V5

V5 is a smaller, clearer replacement for the earlier notebook. It creates HarmBench-style chemistry target tasks, compares four model families under the same assignments, and keeps the exact V4.2 candidate and judgment CSV schemas.

The Colab setup cell expects this GitHub folder to be named `ChemBreak_V5_CLEAN_PACKAGE`. It also searches automatically for `chembreak_pipeline.py` if the folder is renamed later.

## The one setting

In the Colab notebook, change only:

```python
RUN_TYPE = "test"   # or "pilot"
```

| Run type | Controlled assignments | Generated candidates | Blind judgments |
|---|---:|---:|---:|
| `test` | 1 | 4 | 16 |
| `pilot` | 9, one per harmful capability | 36 | 144 |

`test` uses the same real models and the same pipeline as `pilot`. It is only smaller.

## Models

| Family | Checkpoint |
|---|---|
| A, Qwen | `Qwen/Qwen3-8B` |
| B, Mistral | `mistralai/Mistral-7B-Instruct-v0.3` |
| C, Gemma | `google/gemma-2-9b-it` |
| D, Phi | `microsoft/phi-4` |

Every model has the same two roles:

1. Generate one candidate for every controlled assignment.
2. Blind-judge every candidate, including its own candidate.

The judge prompt never includes the generator identity. Cross-family scores exclude a model's self-judgment when candidates are ranked.

## Four-stage workflow

1. Build deterministic assignments from the taxonomy, generation plan, and entity pool.
2. Load each model in turn and generate its candidates.
3. Load each model in turn and judge every candidate.
4. Compare consensus scores, remove near duplicates, and select one provisional task per assignment.

Only one model is kept in GPU memory at a time. Candidate and judgment rows are written immediately, so rerunning the notebook resumes completed work.

The main Colab cell streams model loading, candidate generation, and judgment progress live, including completed counts such as `[3/36]` and `[80/144]`.

## Main outputs

- `candidate_tasks_multimodel.csv`: exact V4.2 candidate schema.
- `judgments_multimodel.csv`: exact V4.2 judgment schema.
- `candidate_consensus.csv`: all-judge and cross-family scores.
- `provisional_task_bank.csv`: the highest-ranked consensus candidate per assignment.
- `generator_comparison.csv`: generator-level comparison.
- `judge_comparison.csv`: judge-level comparison.
- `coverage_report.csv`: coverage by harmful capability.
- `run_summary.json`: final row counts and completeness check.

The notebook downloads the current run directory with Python's archive utility. It checks that the required result files exist before creating the ZIP.

## Colab requirements

- Select a GPU runtime before running the pipeline.
- Accept the Gemma model license on Hugging Face.
- Log in to Hugging Face in the notebook when prompted.

The notebook always uses the real Hugging Face checkpoints. There is no smoke configuration and no mock research run.

## Command-line equivalent

```bash
python chembreak_pipeline.py --run-type test --stage all
python chembreak_pipeline.py --run-type pilot --stage all
```

The generator creates benchmark target requests only. It does not generate answers to those requests.
