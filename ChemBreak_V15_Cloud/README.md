# ChemBreak V15 Cloud

ChemBreak V15 is the Colab Enterprise controller for the ChemBreak chemistry-LLM safety task-bank pipeline.

## Why V15 exists

V14's preflight could falsely mark gpt-oss Judge A as failed even when the endpoint and JSON serialization were working. The smoke-test prompt asked for ten integer scores but did not state the required 0-5 scale, so gpt-oss reasonably returned 0-100 style values. V15 fixes the smoke-test instruction while preserving the strict 0-5 judge contract.

## V15 changes

- Judge A preflight explicitly requires every score to be an integer from 0 to 5 inclusive.
- The harmless smoke test supplies deterministic A and B score arrays to eliminate scale ambiguity.
- The 0-5 parser/validator is unchanged and remains strict.
- Fresh V15 namespace `CBV15C`, run signature, runtime directory, and GCS prefix.

## Preserved production features

- Clean console progress suitable for pilot/production.
- Gemini 3.1 A/B/C batch generation with automatic batch retry and individual fallback.
- Deterministic validation with taxonomy, operation/output, duplicate, leakage, coherence, and entity-domain checks.
- Targeted repair, pre-judge recovery, and refill.
- Concurrent independent gpt-oss-120B Judge A and Gemini 2.5 Pro Judge B.
- Blind Gemini 3.1 adjudication only on genuine disagreement.
- Correct GCS MIME metadata and metadata repair for unchanged objects.
- Resumable checkpoints and run-compatibility guards.

## Runtime layout

Working files:

`/content/chembreak_v15_runtime`

Durable checkpoints:

`gs://rs-foundsecft-mghasemi-default-1/ChemBreak_V15/outputs/<run_type>/`

Run `test` first, then `pilot`, then `production`.
