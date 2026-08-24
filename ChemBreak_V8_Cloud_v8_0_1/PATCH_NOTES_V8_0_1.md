# ChemBreak V8.0.1 Cloud Patch

This patch fixes issues found by the first live Vertex AI preflight.

## Fixes

1. Gemini 3.1 Pro Preview
   - Ignores Gemini 3.x thought parts when extracting the final JSON response.
   - Prevents internal reasoning text from being passed to the JSON parser.

2. Llama 4 Maverick
   - Uses `llama-4-maverick-17b-128e-instruct-maas`.
   - Uses the documented Vertex `v1` chat-completions route.

3. gpt-oss-120B
   - Preflight tests the underlying endpoint only once even though it serves generator, repair, and judge roles.
   - Uses low reasoning during preflight so a 256-token test budget is not consumed entirely by reasoning.
   - Adds exponential retry/backoff for HTTP 429 and transient 5xx errors.
   - Retries an empty reasoning-only response once with lower reasoning.
   - Gives repair and judging larger output budgets.

4. Standard Google Colab authentication
   - Uses the ADC login flow that works in ordinary Colab.
   - Imports `Path`, `subprocess`, `sys`, `json`, `os`, and `shutil` before Drive/Git operations.

No ChemBreak V7 code or outputs are used.
