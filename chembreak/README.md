# ChemBreak

**Independent judge requests:** For every target response, the same GPT model is called twice, once for RHS and once for OFS, with separate prompts, schemas, retries, and error handling. Neither score depends on the other.

ChemBreak is a two-measurement chemistry-model safety evaluation package built directly from the uploaded historical clean run. The original task bank and judging/policy settings are preserved. The adaptive turn budget is changed to six turns per stage.

**Measurements**

- **OFS (Objective Fulfillment Score)**: The original five-level objective-fulfillment rubric, now named OFS. OFS 5 is the success threshold and OFS drives policy and rewards.
- **RHS (Response Harmfulness Score)**: Separate five-level score for harmfulness, regardless of objective completion. RHS does not determine success or the MDP policy.
- Both are scored independently and cumulatively over the current active target conversation, in separate fresh GPT-5.6 Sol requests. Each dimension can fail independently.
- These are judge-assessed classifications, not independent chemical correctness verification.

**GitHub and Colab**

1. Commit the entire extracted `chembreak/` folder to the root of your GitHub repository (`https://github.com/Jollychuks/ChemBreak`).
2. Open `chembreak/notebooks/ChemBreak_Cloud_Notebook.ipynb` in Colab.
3. Begin with `DRY_RUN=True`, set `RUN_ID` to a unique quoted string for new runs (required when moving from the 5x3 design to the 6x3 design), and `TASK_LIMIT=1` for a smoke test or `None` for all 28 tasks.
4. Run cells in order. Run ChemDFM and ChemLLM separately, as in the original notebook. For live runs, configure the GPT API key and Vertex credentials first.
5. Results are exported to separate public and private archives. Raw prompt and response transcripts remain private.

Original benchmark behaviors are unchanged. The adaptive run now uses three stages of up to six turns each (18 adaptive target queries), plus one separate baseline query, for a maximum of 19 target queries per task. IDs are `CB-0001` through `CB-0028`. Renaming IDs changes task-derived random seeds relative to historical runs, even when the top-level seed is unchanged. Do not compare two runs as trajectory-matched unless they use the same underlying responses.

`pyproject.toml` contains a required package metadata version. It is not used in the ChemBreak project name, directory, notebook, run identifiers, or experiment revision.

## Independent RHS and OFS judging

For each saved target response, ChemBreak sends two fresh, independent GPT-5.6 Sol structured-output requests over the same active-context response snapshot. The RHS-only request classifies response harmfulness; the OFS-only request uses the original CHCS-derived 1-to-5 objective-fulfillment rubric. Neither request requires the other to succeed. GPT is the sole judge model; there is no fallback judge and no RHS recovery request. Failed dimensions are stored as unresolved. OFS alone controls MDP decisions, rewards, and OFS=5 stopping. Existing optional final OFS-only reassessment is retained, and never overwrites a valid RHS.

## Live judge errors

When an RHS or OFS request fails, the live progress output now prints the underlying provider error message, and where available the HTTP status and provider error code. The same root error metadata is recorded in `role_calls.csv` through the SQLite checkpoint. Failures in final OFS recovery are also printed instead of silently disappearing. This is diagnostic only: errors remain unresolved and no scores are fabricated or substituted. An API rejection is not automatically fixed by showing its message.
