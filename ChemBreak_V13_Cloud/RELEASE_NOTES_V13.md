# ChemBreak V13 Release Notes

## Why V13 exists

The V12 9-task test exposed two judge-interface failures:

- gpt-oss frequently returned valid JSON with criterion scores directly under each candidate record, while V12 required an extra nested `scores` object.
- Gemini 2.5 Pro repeatedly returned malformed or truncated JSON during the same deeply nested judging task.

V13 fixes the serialization boundary instead of changing the research methodology.

## Changes

- Fresh V13 namespace: `CBV13C-####`.
- Fresh output prefix: `ChemBreak_V13`.
- Flat compact judge JSON with fixed 10-score arrays.
- Robust judge normalization accepts V13 flat output plus common V11/V12 nested/direct variants.
- Gemini 2.5 Pro keeps Vertex structured output with an explicit shallow response schema.
- Gemini 2.5 Pro Judge B thinking budget set to 128 tokens.
- gpt-oss Judge A reasoning effort reduced from high to medium.
- gpt-oss Judge A output budget reduced to 1400 tokens.
- Gemini Judge B output budget reduced to 1200 tokens.
- Adjudicator output budget reduced to 1200 tokens.
- Judge A does not automatically repeat a costly model call after a parseable local contract failure.
- Gemini Judge B retains one compact JSON retry for malformed JSON.
- Preflight now smoke-tests the actual V13 judge serialization contract for both judge models.
- Enterprise working directory moved to `/content/chembreak_v13_runtime`.
- Batched Gemini generation from V12 is retained.

## Unchanged methodology

- three candidates per assignment target
- deterministic validation before LLM judging
- exact-defect repair
- two independent judges
- concurrent Judge A and Judge B calls per assignment
- blind Gemini 3.1 Pro adjudication on disagreement
- refill and single-candidate recovery rules
- 9-task test, 100-task pilot, 500-task production targets
