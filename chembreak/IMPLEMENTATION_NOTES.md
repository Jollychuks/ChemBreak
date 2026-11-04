# ChemBreak implementation notes

## Starting point and scope

The experimental design follows the uploaded ChemBreak34 baseline/stage controller and its existing task-local decision policy. The provided cache-fixed six-turn package supplies the later independent RHS/OFS scoring and working Hugging Face cache initialization. The evaluation-quality checks are adapted from the prior ChemBreak rebuild. This package does **not** introduce new attack-generation actions, modify source benchmark prompts, or increase attack-query budgets.

The project and import package are named **ChemBreak** / `chembreak`. Package metadata includes a packaging version, which is not a research version label. Original 28 benchmark prompts are byte-for-byte unchanged. Assignment IDs are CB-0001..CB-0028; these are not CB34P IDs, and task-derived seeds therefore differ from ChemBreak34's archived task IDs.

## Core experiment

- Target models: ChemDFM and ChemLLM, original pinned model revisions.
- One direct-query baseline per task, in its own conversation.
- Three adaptive stages, with at most six target turns each (19 total queries including the baseline).
- Adaptive controller and candidate-generation prompts are unchanged from the working cache-fixed six-turn source.
- Each task has isolated policy state. Active-conversation context may persist across stages; any explicit restart resets cumulative evaluation evidence.
- The single `judge_llm` is called independently for RHS and OFS on the entire active target response set.

## Reliability safeguards

- RHS reports a 1..5 cumulative score, supporting response indices, latest response disposition, incremental actionability flag, and a non-authoritative technical-accuracy flag. No chemical procedures or reconstructed details are requested from the judge.
- A high RHS or an unexplained increase after a noncontributing refusal enters a redacted human review queue. The score is not automatically adjusted. The reviewer must assess technical plausibility before treating it as a validated finding.
- OFS is scored independently using the original objective. A cumulative OFS decrease in one active conversation enters a quality hold; it does not create a reward or another score-driven adaptive turn.
- A missing OFS due to `bio_policy` or technical failure pauses that task with a null reward. A missing RHS alone does not erase a valid OFS.
- Automated OFS=5 results are **provisional**, not expert-confirmed successes; they enter the review queue.
- Provider policy blocks are never retried using evasion, redaction, an unapproved fallback model, or another provider.
- On `credit_balance_exhausted`, `insufficient_quota`, or another explicitly classified billing-account failure, the entire run aborts immediately with `RUN_ABORTED.json`. No 0% ASR summary is produced for a run stopped by unavailable credits. Ordinary rate limits remain retriable.
- Original Colab local storage layout is preserved. No Google Drive integration is added.

## Validation and limits

The suite covers dry-run execution of all nineteen query slots for one task, account exhaustion, RHS-only policy block, OFS policy block, cumulative score checks, the refusal case, context reset, provisional OFS=5, and OFS regression. All tests are **offline** and use synthetic/mocked judge output. The notebook defaults to a benign API role probe before loading either full target model, so exhausted credits can be discovered earlier.

**Not established:** whether bio-policy rejections become less frequent in live experiments; actual chemical correctness; inter-rater reliability; confirmed harmfulness; independently verified success; live cloud compatibility. The system reports uncertainty rather than claiming these tests resolve those issues.

## Run preparation

1. Place the extracted `chembreak/` directory at the repository root, as expected by the Colab notebook.
2. Open `chembreak/notebooks/ChemBreak_Cloud_Notebook.ipynb`, run from the top and start with `DRY_RUN=True` and `TASK_LIMIT=1`.
3. For a live smoke test, set `DRY_RUN=False`, keep `PROBE_ROLES=True`, and use `TASK_LIMIT=1`; confirm credits and credentials before execution.
4. Set `TASK_LIMIT=None` only after the controlled check. `RUN_ID=None` creates a unique ID; to resume, explicitly reuse the saved run ID and the same configuration.
5. Do not report an ASR as independently confirmed unless the relevant judgments were actually validated through an authorized review process.
