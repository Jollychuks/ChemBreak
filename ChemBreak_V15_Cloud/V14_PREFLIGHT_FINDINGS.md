# V14 Preflight Finding Used for V15

The V14 judge-contract preflight correctly reached gpt-oss-120B but its harmless smoke-test prompt did not state the benchmark score range. Judge A therefore returned 0-100 style values such as 85 and 90, while the V14 normalizer correctly enforced the intended 0-5 scale and marked the contract test as failed.

V15 fixes the smoke test rather than weakening the validator. The preflight now explicitly requires exactly ten integer scores per candidate, each between 0 and 5 inclusive, and supplies deterministic example arrays for the harmless serialization test.

The production judge contract remains on the same 0-5 scale. No endpoint, model, taxonomy, judging architecture, or benchmark-quality threshold is changed by this fix.
