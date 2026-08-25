# ChemBreak V15 Design Lock

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

## V15 preflight rule

The judge-contract smoke test must exercise the same flat serialization shape and 0-5 score range used by the real judging stage. It must never weaken the 0-5 validator to accommodate a smoke-test ambiguity. Judge A receives deterministic harmless example arrays `[5,...,5]` and `[4,...,4]`; Judge B receives the same textual contract plus its Vertex response schema.

## Reliability inherited from V14

1. Initial generation uses one A/B/C batch call per assignment.
2. A malformed batch response gets one automatic structured retry.
3. If the batch still fails, only missing candidate slots are generated with smaller individual calls.
4. Successful work is checkpointed and reused.
5. Transport/API/serialization failures are not benchmark votes.
6. Validation includes entity-to-hazard-domain consistency.
7. GCS uploads set and repair MIME metadata.
8. Clean console mode shows compact progress plus retries, fallbacks, failures, long waits, and stage summaries.
