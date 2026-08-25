# ChemBreak V14 Release Notes

V14 is a focused reliability and production-usability release built from the successful V13 architecture.

## Fixed from the V13 test

- Initial Gemini 3.1 batch JSON can now retry once automatically.
- Persistent batch JSON failure falls back to individual candidate generation for only that assignment.
- Known chemical-warfare entities such as Sarin/VX are no longer broadly eligible for HD3 merely because the ChemSafety source groups weapons and poisons together.
- Deterministic validation now checks entity-to-hazard-domain consistency.
- Cloud Storage uploads explicitly set MIME metadata, including `text/csv; charset=utf-8` for CSV.
- The notebook's bucket preflight verifies CSV MIME metadata.
- Production console output is substantially cleaner.
- Full call-level diagnostics are retained in `technical_events.jsonl`.
- Recovery loops stop once the target number of assignments is selected, avoiding redundant empty cycles.

## Preserved from V13

- Gemini 3.1 batch generation for speed
- deterministic validation and targeted repair
- gpt-oss-120B Judge A at medium reasoning
- Gemini 2.5 Pro Judge B flat response schema
- concurrent independent judging
- blind Gemini 3.1 adjudication
- GCS checkpoint/resume
- fresh namespace and run signature
