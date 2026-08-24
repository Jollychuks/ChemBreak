# ChemBreak V11 Cloud

ChemBreak V11 is the quality-first task-bank construction pipeline developed after reviewing the V9 and V10 test outputs.

## Locked V11 model roles

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

Llama is not used in V11.

## Main methodological update

V10 used three different generator families. V11 uses one stronger generator, Gemini 3.1 Pro, to create three deliberately different candidate requests for every assignment. Diversity is controlled through candidate language profiles, request forms, sentence structure, opening patterns, and global similarity checks instead of relying on model-family variation.

Every active candidate set is evaluated by two independent judges concurrently. Judge A and Judge B do not see generator identities or each other's decisions. If they agree on a winner, the task is selected. If they both reject all candidates, the assignment enters refill. If they disagree, Gemini 3.1 Pro adjudicates without seeing model identities.

## Candidate-count logic

- 3 valid candidates: both judges evaluate all 3.
- 2 valid candidates: both judges evaluate both.
- 1 valid candidate: V11 makes up to two targeted pre-judge refill attempts. If only one candidate remains, both judges independently qualify or reject the single candidate.
- 0 valid candidates: the assignment enters full refill.

A technical judge failure is never treated as a vote. Successful judge results are checkpointed, and rerunning the judge stage retries only the missing judge for the same candidate set.

## Run sizes

- `test`: 9 final tasks
- `pilot`: 100 final tasks plus 15 reserve assignments
- `production`: 500 final tasks plus 75 reserve assignments

## Fresh provenance

V11 does not use generated V8, V9, or V10 tasks as generation input. It bootstraps fresh source snapshots from:

- IDEA-XL ChemSafety `substances.json` for chemical/entity provenance.
- HarmBench `chemical_biological` behaviors only as an external similarity reference.

The source snapshots are SHA-256 recorded in `source_snapshot_manifest.json`.

## Main pipeline

`preflight -> bootstrap -> plan -> generate -> validate -> repair -> prejudge_refill -> judge -> adjudicate -> refill/recovery -> finalize`

The `all` stage executes the same sequence automatically and repeats the recovery cycle up to the configured refill limit.

## Colab

Use `ChemBreak_V11_Google_Colab.ipynb` for standard Colab. The notebook streams subprocess output line by line, displays text progress bars, emits 20-second heartbeats during long model calls, and stores checkpoints in Google Drive by default.

A local Colab GPU is not required because Gemini and gpt-oss inference is performed remotely through Vertex AI.

## Fresh namespace

V11 uses:

- project folder: `ChemBreak_V11_Cloud/`
- assignment namespace: `CBV11C-####`
- Drive root: `MyDrive/ChemBreak_V11/`
- assignment file: `assignments_v11.csv`
- entity file: `entities_v11.csv`
- reference file: `external_reference_behaviors_v11.csv`

V11 refuses to resume into an incompatible checkpoint when the pipeline, prompts, taxonomy, model configuration, seed, or run type changed.
