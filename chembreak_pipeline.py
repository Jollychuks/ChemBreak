# ChemBreak V7.0 validation report

Validation date: 2026-08-18

## Result

The local offline validation suite passed all 10 tests.

## Verified contracts

- Python modules compile successfully.
- The Colab notebook is valid notebook JSON and every ordinary Python code
  cell parses successfully.
- The model registry contains two generators and one Phi-4 judge.
- The test profile creates 1 assignment.
- The pilot profile creates 29 assignments, one for each plan row.
- The production profile creates 550 initial assignments for a 500-task final
  target and a 50-assignment reserve.
- `production_quotas.csv` contains 29 rows whose targets total 500 and reserves
  total 50.
- The plan covers all 9 harmful capabilities, 8 hazard domains, and 14 output
  types.
- All 12 request-form families occur in the pilot plan.
- Natural help, how, and what request forms pass their compatible validators.
- OT6 accepts a supplied input quantity but rejects a disclosed requested
  result.
- OT14 requires a supplied self-contained claim to validate.
- Safe response normalization removes wrappers without rewriting chemical
  content.
- Critical Phi-4 defect flags enforce disqualification and score caps.
- An offline end-to-end test generates, validates, judges, selects, finalizes,
  and resumes without duplicating saved work.
- A full 29-assignment offline pilot exercises every plan row, all 14 output
  types, all 12 request-form families, and one grouped load for each model role.
- A two-cycle failure test sends the exact prior error to the next retry and
  permits independent Phi-4 qualification of the valid survivor only after the
  partner exhausts both complete generation cycles.

## Scope

The automated suite uses deterministic fake model runners. It does not download
or execute the three large Hugging Face checkpoints. Run the notebook's
preflight cell and the one-assignment `test` profile on Colab before starting
the pilot or production profile.
