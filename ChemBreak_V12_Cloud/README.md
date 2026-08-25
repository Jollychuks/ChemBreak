# ChemBreak V12 Cloud

ChemBreak V12 is the Enterprise-first, quality-first ChemBreak task-bank construction pipeline.

V12 keeps the V11 research design while fixing the Gemini 2.5 Pro structured-judgment failure and reducing runtime without weakening the two-independent-judge methodology.

## Locked model roles

| Function | System |
|---|---|
| Generation | Gemini 3.1 Pro Preview |
| Repair | Gemini 3.1 Pro Preview |
| Pre-judge refill | Gemini 3.1 Pro Preview |
| Full refill | Gemini 3.1 Pro Preview |
| Deterministic validator | Python rules |
| Judge A | gpt-oss-120B |
| Judge B | Gemini 2.5 Pro |
| Adjudicator | Gemini 3.1 Pro Preview |

Llama is not used.

## V12 reliability fix

Gemini 2.5 Pro Judge B now uses Vertex AI controlled JSON generation with an explicit response schema. A technical judge failure remains technical pending and is never treated as a vote.

## V12 runtime improvements

- Initial generation uses one Gemini 3.1 structured batch call per assignment for A/B/C instead of three separate calls.
- Gemini generation, repair, refill, Judge B, and adjudication use explicit response schemas where applicable.
- The JSON-format fallback retry budget is reduced because the response shape is constrained by Vertex AI.
- Inter-call pacing is reduced while transport/API backoff remains in place.
- Judge A and Judge B still execute concurrently for each assignment.

## Candidate-count logic

- 3 valid candidates: both judges evaluate all 3.
- 2 valid candidates: both judges evaluate both.
- 1 valid candidate: targeted pre-judge refill is attempted first; if still alone, both judges independently qualify or reject the single candidate.
- 0 valid candidates: full refill.

## Run sizes

- test: 9 final tasks
- pilot: 100 final tasks plus 15 reserve assignments
- production: 500 final tasks plus 75 reserve assignments

## Provenance

V12 does not use prior generated ChemBreak tasks as generation input. It bootstraps fresh source snapshots from IDEA-XL ChemSafety `substances.json` for entity provenance and HarmBench `chemical_biological` behaviors only as an external similarity reference.

## Pipeline

`preflight -> bootstrap -> plan -> generate -> validate -> repair -> prejudge_refill -> judge -> adjudicate -> refill/recovery -> finalize`

## Enterprise deployment

Use only `ChemBreak_V12_Colab_Enterprise.ipynb`.

The default durable checkpoint location is:

`gs://rs-foundsecft-mghasemi-default-1/ChemBreak_V12/outputs/<run_type>/`

The notebook verifies object-level read/write/delete access and does not attempt to create a bucket.

## Fresh namespace

- project folder: `ChemBreak_V12_Cloud/`
- assignment namespace: `CBV12C-####`
- assignment file: `assignments_v12.csv`
- entity file: `entities_v12.csv`
- reference file: `external_reference_behaviors_v12.csv`
- durable prefix: `ChemBreak_V12/`

V12 refuses to resume into an incompatible checkpoint when the pipeline, prompts, taxonomy, model configuration, seed, or run type has materially changed.
