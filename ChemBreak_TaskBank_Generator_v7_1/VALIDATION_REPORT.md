# ChemBreak V7.1 validation report

Validation date: 2026-08-19

## Result

The local offline validation suite passed all 20 tests.

## Verified contracts

- Python modules compile successfully.
- The Colab notebook is valid notebook JSON and every Python code cell parses.
- The model registry contains Qwen and Ministral as generators and Phi-4 as
  the independent blind judge.
- V7.1 uses a fresh namespace, prompt versions, experiment prefix, output
  paths, and run signature.
- The test profile creates 1 assignment.
- The pilot profile creates 29 assignments, one for every production plan row.
- The production profile creates 550 initial assignments for a 500-task final
  target and a 50-assignment reserve.
- The 29 explicit production targets total 500 and the reserves total 50.
- The plan covers all 9 harmful capabilities, 8 hazard domains, 14 output
  types, and 12 request-form families.
- Candidate validation returns every deterministic defect found in one pass.
- Case-insensitive entity matching restores the controlled capitalization in
  saved fields.
- Safe normalization removes code fences, model end tokens, whitespace, and
  paired Markdown emphasis without rewriting substantive content.
- Operation-aware field consistency accepts natural paraphrases while
  rejecting changed objectives.
- OT5 accepts the required grammatical comparison forms, rejects vague sets,
  permits genuine competing inputs, and classifies named answer examples as
  leakage.
- OT6 supplied calculation inputs, OT7 observed diagnostic quantities, OT9
  unified separation verbs, OT13 scale-up requests, and OT14 supplied claims
  follow their output-specific rules.
- Internal taxonomy identifiers and generic harm labels are rejected.
- Critical Phi-4 defect flags enforce disqualification and score caps.
- All-perfect judge scores require a flawless declaration and no material
  weaknesses.
- Tied qualified pairs are rejudged in reversed order and positional results
  are written to a dedicated report.
- Phi-4 defects are sent back to the appropriate generator in grouped repairs
  with the rejected response and exact correction categories.
- Missing partners receive grouped repair before a survivor may be judged
  alone under stricter thresholds.
- Production session chunks save progress and do not create refill assignments
  while existing generation, repair, or judgment work remains.
- Atomic report writes replace temporary files correctly, and resume testing
  does not duplicate completed candidates or judgments.
- The full 29-assignment offline pilot completes all plan, output-type, and
  request-form coverage checks.

## Scope

The automated suite uses deterministic fake model runners. It does not download
or execute the three large Hugging Face checkpoints. Run the notebook preflight
cell and the one-assignment test profile on Colab before starting the pilot or
production profile.
