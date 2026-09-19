# ChemBreak20 v20.0.0

- Replaces exact-prompt replay with abstract route memory.
- Adds immutable source-derived task locks.
- Adds a pre-target goal-preservation gate.
- Removes an independent Judge-LLM success boolean; final success is deterministic Python.
- Adds response-correctness scoring to the final gate.
- Adds target-specific independent controllers for ChemDFM and ChemLLM.
- Uses deterministic seeded shuffles for the three online-learning orders and records each order in provenance.
- Adds crash-safe rollback of partially completed learning episodes.
- Preserves provider-policy-block continuity without consuming a target query.
- Adds frozen terminal exploitation with two route-informed attempts followed by two distinct frozen-policy syntheses.
- Adds terminal ASR@1..4 and cumulative discovery ASR@1/@4/@8/@12/@16.
- Includes an optional equal-budget non-adaptive control, disabled by default.

- Uses target-native dialogue templates for ChemDFM and ChemLLM instead of a single generic formatter.
- Adds non-policy Attack-LLM/Judge error isolation and target-generation error logging so one technical failure does not crash the entire target run.
- Detects explicit OpenAI Responses-API refusal items as provider blocks.
