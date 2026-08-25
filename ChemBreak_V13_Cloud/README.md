# ChemBreak V13 Cloud

ChemBreak V13 is the Colab Enterprise controller for building a controlled chemistry LLM safety task bank.

## V13 focus

V13 preserves the V11/V12 research design and fixes the judging interface that failed during the V12 test. The two independent judges remain gpt-oss-120B and Gemini 2.5 Pro, run concurrently for each active candidate set. Gemini 3.1 Pro remains the generator, repair/refill model, and blind adjudicator.

The V13 judge contract is intentionally shallow and compact:

- fixed 10-score arrays per candidate
- one qualification boolean per candidate
- one short issue string per candidate
- one selection field
- one short reason

The parser accepts the V13 flat contract and the common nested/direct score shapes observed in V11/V12.

## Runtime improvements

- gpt-oss Judge A reasoning effort reduced from high to medium
- Judge A JSON model retry disabled for parseable contract failures
- Gemini 2.5 Pro uses a minimal thinking budget of 128 tokens for Judge B
- smaller judge and adjudicator output budgets
- batched three-candidate Gemini generation retained from V12
- two judges still run concurrently per assignment
- working files use `/content/chembreak_v13_runtime` rather than the smaller root disk

## Durable outputs

The Enterprise notebook writes durable checkpoints to:

`gs://rs-foundsecft-mghasemi-default-1/ChemBreak_V13/outputs/<run_type>/`

Use `test` first, then `pilot`, then `production`.
