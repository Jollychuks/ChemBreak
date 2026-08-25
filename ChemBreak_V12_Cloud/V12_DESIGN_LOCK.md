# ChemBreak V12 Design Lock

## Model roles

- Generator: Gemini 3.1 Pro Preview
- Repair: Gemini 3.1 Pro Preview
- Refill: Gemini 3.1 Pro Preview
- Deterministic validator: Python
- Judge A: gpt-oss-120B
- Judge B: Gemini 2.5 Pro
- Adjudicator: Gemini 3.1 Pro Preview, blind to model identities

## Candidate logic

- Target three candidates per assignment.
- Initial A/B/C generation is one structured Gemini batch call per assignment.
- Three valid candidates: both judges score all three.
- Two valid candidates: both judges score both.
- One valid candidate: targeted pre-judge refill attempts first, then dual single-candidate qualification if still alone.
- Zero valid candidates: full refill.

## Judging

Both judges independently evaluate every active candidate set. They do not see generator identity or the other judge's result. Agreement selects or rejects. Disagreement routes to blind Gemini 3.1 adjudication. A parser/API failure remains technical pending and is never treated as a vote.

Gemini 2.5 Pro structured judgment uses Vertex AI controlled generation with an explicit response schema.

## Runtime design

- Colab Enterprise is the only supported notebook in this release.
- Durable checkpoints use the existing project bucket configured in the Enterprise notebook.
- Live progress, heartbeat, elapsed time, ETA, and resumability remain required.
- V12 uses a fresh namespace, fresh output prefix, and run signature. V11 outputs are never mixed into V12.
