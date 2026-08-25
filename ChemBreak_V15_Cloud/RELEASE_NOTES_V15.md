# ChemBreak V15 Release Notes

V15 is a focused preflight-correctness release built on the V14 production architecture.

## Fixed from V14

- Judge A preflight now explicitly requires the intended 0-5 scoring scale.
- The harmless serialization smoke test supplies deterministic score arrays so gpt-oss cannot reasonably infer a 0-100 scale.
- The validator remains strict: out-of-range scores are still rejected.
- V15 uses a fresh namespace, run signature, local runtime directory, and Cloud Storage prefix.

## Preserved from V14

- clean production console output with detailed diagnostics in `technical_events.jsonl`
- one-call Gemini 3.1 A/B/C generation with one batch retry and individual fallback
- deterministic validation, including entity-to-hazard-domain consistency
- targeted repair and pre-judge recovery
- concurrent independent gpt-oss-120B and Gemini 2.5 Pro judging
- blind Gemini 3.1 Pro adjudication on genuine disagreement
- proper Cloud Storage MIME metadata, including `text/csv; charset=utf-8`
- GCS checkpoint/resume and early recovery-loop stop after target completion

V15 does not use V14 generated tasks as generation input.
