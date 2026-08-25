# ChemBreak V14 Cloud

ChemBreak V14 is the Colab Enterprise controller for building the ChemBreak chemistry-LLM safety task bank.

## Why V14 exists

V13 proved the two-independent-judge architecture works in Colab Enterprise and completed the 9-task test, but it also exposed four issues that matter before the 100-task pilot and 500-task production run:

1. Gemini 3.1 batch generation was fast but only 5/9 assignments completed on the first batch pass because four responses contained malformed JSON.
2. Recovery eventually filled all nine assignments, but using full refill for transport/serialization failures is inefficient.
3. The source-domain mapping allowed a chemical-warfare entity to be assigned to the poison/TIC domain.
4. Notebook output was too verbose for production-scale runs, and CSV objects were uploaded to Cloud Storage without an explicit `text/csv` MIME type.

## V14 changes

- **Fast generation retained:** one Gemini 3.1 batch call normally returns A/B/C for an assignment.
- **Automatic batch recovery:** malformed batch JSON receives one structured retry.
- **Individual fallback:** if the batch retry still fails, V14 generates only the missing A/B/C slots with smaller structured calls instead of waiting for full refill.
- **Entity-domain consistency:** ChemSafety `Chemical Weapons and Poisons` entries are mapped more carefully, with strong entity-domain overrides and a deterministic validator check.
- **Clean console mode:** notebook output shows stage start/completion, compact progress, retries, failures, recovery events, and long waits. Full model-call diagnostics go to `technical_events.jsonl`.
- **Production-scaled progress:** large stages emit at most about 25 routine progress updates; failures and retries are always shown immediately.
- **Correct Cloud Storage metadata:** CSV files upload as `text/csv; charset=utf-8`; JSON, JSONL, Markdown, images, and archives receive matching MIME types.
- **MIME preflight:** the Enterprise notebook verifies a temporary CSV object is stored as `text/csv` before the run.
- **V13 judge architecture preserved:** Judge A is gpt-oss-120B at medium reasoning, Judge B is Gemini 2.5 Pro with the flat structured contract, and both run concurrently and independently.

## Runtime layout

Working files:

`/content/chembreak_v14_runtime`

Durable checkpoints:

`gs://rs-foundsecft-mghasemi-default-1/ChemBreak_V14/outputs/<run_type>/`

Run `test` first, then `pilot`, then `production`.
