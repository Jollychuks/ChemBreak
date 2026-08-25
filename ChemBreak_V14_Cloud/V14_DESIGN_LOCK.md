# ChemBreak V14 Design Lock

## Research architecture preserved

Taxonomy/matrix assignment -> Gemini 3.1 generation -> deterministic validation -> targeted repair -> pre-judge recovery -> independent concurrent Judge A and Judge B -> blind adjudication on genuine disagreement -> full refill only when needed -> final task bank and metrics.

## Models

- Generator: Gemini 3.1 Pro Preview
- Repair/refill: Gemini 3.1 Pro Preview
- Judge A: gpt-oss-120B MaaS, medium reasoning
- Judge B: Gemini 2.5 Pro, flat structured response contract
- Adjudicator: Gemini 3.1 Pro Preview

## Run sizes

- Test: 9
- Pilot: 100 plus 15 reserve
- Production: 500 plus 75 reserve

## V14 reliability rules

1. Initial generation uses one A/B/C batch call per assignment.
2. A malformed batch response gets one automatic structured retry.
3. If the batch still fails, only the missing candidate slots are generated with smaller individual calls.
4. Successful candidates are checkpointed immediately and never intentionally regenerated.
5. Transport/API failures and structured-output failures are logged separately from benchmark-quality rejection.
6. Full refill is for assignments with no valid pool or genuine judge rejection, not the first response to an initial serialization problem.

## V14 validation additions

- Preserve all existing deterministic checks from V13.
- Use source-level `allowed_hazard_domains` during planning.
- Strong entity overrides prevent known controlled-drug, explosive, chemical-warfare, poison, environmental, contamination, industrial-sabotage, and cross-domain system targets from being assigned to incompatible hazard domains.
- Validation emits `hazard_domain_entity_mismatch` when an assignment contradicts the source/semantic entity-domain mapping.

## Console policy

Default `console_mode = clean`.

Show:
- stage start and completion
- compact progress bar/counters
- retries
- fallbacks
- failures
- recovery events
- long waits
- final summary

Hide from normal console:
- every model CALL START/CALL DONE line
- routine checkpoint-mirror chatter
- raw response fragments

Preserve detailed model call timing and errors in `technical_events.jsonl` and `errors.jsonl`.
