# ChemBreak V11 Release Notes

## Major changes from V10

- Replaced three-generator bulk generation with Gemini 3.1 Pro as the sole generator.
- Retained three candidate slots per assignment through controlled language profiles.
- Moved repair from gpt-oss-120B to Gemini 3.1 Pro.
- Removed Llama 4 entirely.
- Changed judging to two independent concurrent judges for every active candidate set.
- Judge A: gpt-oss-120B.
- Judge B: Gemini 2.5 Pro.
- Both judges evaluate all surviving candidates rather than only a preselected pair.
- Added explicit harmful-intent fidelity criteria so judges do not prefer sanitized tasks.
- Added structured-output retries to address malformed judge JSON.
- Added targeted pre-judge refill when only one candidate survives.
- Added dual single-candidate qualification after pre-judge refill is exhausted.
- Added full refill with two fresh Gemini candidates per cycle and complete failure history.
- Added refill-candidate repair using exact deterministic defects.
- Preserved blind Gemini 3.1 Pro adjudication for genuine judge disagreement.
- Added explicit technical-pending behavior. A judge API/parser failure is never treated as a vote.
- Added concurrent judge execution with `ThreadPoolExecutor`.
- Added visual text progress bars in addition to heartbeats, percentages, elapsed time, and ETA.
- Added V11-specific source, plan, model, and checkpoint namespaces.
- Expanded final metrics for validation, repair, refill, judge agreement, adjudication, errors, coverage, and diversity.
