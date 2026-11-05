# ChemBreak Gemma 4 rebuild notes

- Replaced the GPT Responses API evaluator with `LocalGemmaJudgeLLM` serving `google/gemma-4-26B-A4B-it` using vLLM's localhost chat-completions JSON schema endpoint. GPT API credentials and client dependency are no longer required.
- Kept RHS ordinal 1–5, changed OFS to 1–10, and configured provisional OFS=10 as the success condition. Preserved cumulative active-context measurements with separate calls.
- Added deterministic `evidence.py` for short exact-match citations, source turn attribution, SHA256 response and citation hashes, and verified character offsets.
- Added Python score/class/reason checks, cumulative score regression flags, invalid evidence holds, high-score independent-validation flags and strong RHS/OFS conflict review.
- Updated state bands, policy thresholds, reward normalization, reporting counters, metadata, summary definitions, and review queue so the new ordinal OFS range is used consistently. No new attack strategies were introduced.
- Updated Cloud Notebook to load the uploaded rebuilt ZIP by default rather than silently reverting to the old GitHub branch. Live preflight probes local Gemma `/v1/models` and requires no OpenAI API key.
- Added Gemma vLLM startup script, harmless local probe and two mock-only test suites.

All tests were executed without a live Gemma GPU or Vertex API. **No live reliability claim has been made.** Offsets validate provenance only; scientific correctness requires independent human review. Do not mix old 1–5 OFS scores or old checkpoints with new 1–10 OFS results.
