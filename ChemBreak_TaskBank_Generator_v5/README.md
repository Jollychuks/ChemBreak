# ChemBreak Task Bank Generator V5

V5 is a smaller task-bank pipeline built around HarmBench-style base behaviors.

It keeps the four-family comparison requested for ChemBreak:

- A, B, C, and D generate from the same controlled assignments.
- A, B, C, and D blindly judge every candidate.
- Python selects one provisional winner per assignment.
- A human reviewer makes the final inclusion decision.

## What changed from V4.2

V5 removes the scenario engine, the 271-row factorial matrix, the 15-item generator checklist, GitHub checkpoint code, and environment-freeze code. It preserves the V4 candidate and judgment output schemas.

The generator returns the same four content fields used in V4:

```json
{
  "benchmark_prompt": "one direct harmful chemistry request",
  "main_goal": "short harmful objective",
  "chemical_entity": "controlled entity",
  "distinctive_dimension": "main distinguishing feature"
}
```

Python controls the taxonomy, chemical entity assignment, identifiers, seeds, validation, deduplication, ranking, and V4-compatible output schema.

## Files

- `ChemBreak_TaskBank_Generator_V5.ipynb`: Colab entry point.
- `chembreak_v5.py`: complete generation, judging, selection, and promotion pipeline.
- `config.json`: pilot or full run settings.
- `config_smoke.json`: fast offline structural test.
- `generation_plan.csv`: curated HC, HD, and OT assignments.
- `entity_pool.csv`: controlled chemical entities and scope rules.
- `taxonomy.json`: nine harmful capabilities, eight hazard domains, and fourteen output types.
- `reference_harmbench.csv`: reference behaviors used only for copy detection.
- `generator_prompt.txt`: concise base-task authoring prompt.
- `judge_prompt.txt`: concise blind scoring prompt.
- `models.json`: the four registered checkpoints.

## Output format

`candidate_tasks_multimodel.csv` uses the same ordered columns as V4, from `experiment_id` through `generated_at_utc`.

`judgments_multimodel.csv` also preserves the V4 structure, including the eight score columns, `overall_quality_score`, `validator_decision`, and `judge_reason`.

Because V5 does not use the scenario engine, `allowed_scenarios` and `selected_scenarios` are recorded as `NONE`, while `scenario_plan_version` is recorded as `NOT_USED`.

## Run the offline smoke test

```bash
python chembreak_v5.py --config config_smoke.json --stage all --backend mock
```

Expected output:

- 9 controlled assignments, one for each HC category
- 36 candidates, four per assignment
- 144 judgments, four per candidate
- 9 provisional task-bank rows

## Run the pilot on Hugging Face models

Use a Colab GPU runtime, authenticate with Hugging Face when required, then run:

```bash
pip install -r requirements_colab.txt
python chembreak_v5.py --config config.json --stage all --backend hf
```

Models load sequentially and are released before the next family loads.

The Gemma checkpoint requires accepting its Hugging Face access conditions. If it is unavailable, remove family `C` from both family lists or replace its model ID with another approved Gemma checkpoint.

## Expand after the pilot

After reviewing the pilot:

1. Change `mode` from `pilot` to `full`.
2. Increase `tasks_per_plan_row` from 1 to 2 or 3.
3. Keep the same assignment plan across every generator family.
4. Run generation, judging, and finalization again under a new `experiment_id` and output directory.

## Human review and final promotion

Automated acceptance creates `human_review_queue.csv`, not final ground truth.

Fill `HumanDecision` with `ACCEPT`, `REVISE`, or `REJECT`. Then run:

```bash
python chembreak_v5.py --config config.json --stage promote
```

Only human-accepted rows are written to `final_task_bank.csv`.

## Main outputs

- `assignments.csv`
- `candidate_tasks_multimodel.csv`
- `judgments_multimodel.csv`
- `candidate_consensus.csv`
- `provisional_task_bank.csv`
- `human_review_queue.csv`
- `generator_summary.csv`
- `judge_summary.csv`
- `coverage_report.csv`

## Research interpretation

Family labels refer to the registered checkpoints, not every model released by the corresponding developer. Report results as a comparison of those checkpoints under the same V5 configuration.
