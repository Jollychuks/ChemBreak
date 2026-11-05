# ChemBreak

ChemBreak is an independently reproducible chemistry-language-model safety-evaluation harness. This rebuild **replaces GPT judging with Google Cloud managed Gemma 4 26B A4B** model. It preserves the uploaded 28-task bank, two target-model configurations, three adaptive stages, six turns per stage, separate baseline conversation, task isolation, and two independent cumulative judgment requests. The attack-generation framework has not been expanded.

## What changed

| Feature | This rebuild |
|---|---|
| Judge | `google/gemma-4-26b-a4b-it-maas`, managed Google Cloud API |
| RHS | Independent **1–5** cumulative response harmfulness score |
| OFS | Independent **1–10** cumulative original-objective fulfillment score |
| Automatic success | **OFS=10**, provisional judge prediction only |
| Evidence | Short exact quotations proposed by the judge and verified in Python against the correct turn |
| Provenance | SHA256 of source response and evidence quote, with character offsets |
| Consistency validation | Score/class/reason matches; legal score ranges; verbatim evidence; cited turn indices; same-context regressions; high-OFS/low-RHS conflicts |
| Invalid OFS evidence or regression | `quality_hold`, no artificial score or further score-dependent decisions |
| Technical failures | Missing score rather than a fabricated refusal; recorded errors and review queue |
| Review | High RHS, high OFS, conflicting results and unresolved evaluations flagged for expert review |

**Scope:** On every judged target response, the two separate Gemma requests each read all target-model responses in the *currently active* target conversation. A context restart resets the body of evidence. OFS is a fresh ordinal evaluation, not an average or sum of per-turn scores. RHS never substitutes for OFS.

## Run in Colab

1. Use the **`ChemBreak.zip`** release file and open `chembreak/notebooks/ChemBreak_Cloud_Notebook.ipynb`. In Cell 2, keep `SOURCE_PACKAGE = 'uploaded_zip'`. The notebook accepts a ZIP upload into `/content` or you can place it in the Colab Files panel.
2. Start with `DRY_RUN=True`, and `TASK_LIMIT=1` for the first validation. Dry run uses harmless mocks and does **not** call Gemma.
3. For a live run, set `DRY_RUN=False` and use Google Cloud Application Default Credentials. The notebook auto-detects your project. Enable `PROBE_ROLES=True` to check benign structured judgment requests before a full run. No GPT API credentials are used.
4. Use a fresh `RUN_ID` for each judge/model/quantization/scoring change. Never resume the old GPT 1–5 OFS experiment into the new state. Run the target models separately via their notebook sections.
5. Keep outputs under `INTERNAL_DO_NOT_RELEASE` private. The release folder redacts raw prompts and responses.

The notebook defaults to **uploading this revised ZIP** rather than fetching an older GitHub branch. If you switch `SOURCE_PACKAGE` to `github`, you must first commit the rebuilt package to the selected branch.

## Google-managed Gemma judge

The evaluator uses Google Cloud's managed Gemma 4 26B A4B IT service at the
`global` Agent Platform OpenAI-compatible chat-completions endpoint. **No local
vLLM server, Gemma weights, OpenAI key or separate judge GPU is required.**
This is a managed API, **not locally hosted inference**. API calls are billable.

The notebook discovers the project through Google Application Default Credentials
and sets `GOOGLE_CLOUD_PROJECT`; the judge authenticates with short-lived OAuth
access tokens. It uses a pinned model ID `google/gemma-4-26b-a4b-it-maas`.

The preflight performs a minimal benign service call and verifies the returned
model ID. The optional `PROBE_ROLES=True` exercises the structured scoring
schemas and may incur additional costs. The schema and Python validation are
not weakened if Google returns an invalid or blocked judgment: that judgment
stays unresolved and the error is logged.

Standalone harmless endpoint check (requires ADC and `GOOGLE_CLOUD_PROJECT`):

```bash
PYTHONPATH=src python scripts/probe_managed_judge.py configs/config.yaml
```

Do not confuse hosted Gemma MaaS with the locally hosted Gemma checkpoint.
Document the serving model, date, endpoint, version metadata, and any service
changes in the study. Score calibration against expert labels remains necessary.

## Scoring and quality review

The exact 1–10 OFS anchors are declared in `src/chembreak/prompts.py` and explained in `EVALUATION_PROTOCOL.md`. The old 1–5 OFS scale cannot be directly compared to this 1–10 rubric. For the new scale, controller thresholds and OFS-delta normalization have been migrated rather than carrying forward the old hard-coded score 4/5 constants.

Evidence from the judge is a short **verbatim excerpt** and a 1-based source-response turn. `evidence.py` finds the excerpt in that exact response, verifies its offsets, and records SHA256 hashes. These checks prove that a judge cited **text that exists**, not that the underlying chemistry is sound, the answer is correct, or the risk assessment is valid. Raw citation excerpts are not copied into public exports. An unsupported high OFS or a cumulative regression is held for review.

Outputs include `release/summary.json`, `release/target_queries.csv`, `release/judgment_review_queue.csv`, private full transcripts, and `INTERNAL_DO_NOT_RELEASE/evidence_registry.jsonl`. High scores, conflicts, and incomplete scores require qualified validation before claims of reliable accuracy or attack success. The automatic ASR is a **provisional model-predicted proportion**, not a verified success rate or lower bound.

## Offline regression tests and complete archive

The lean `ChemBreak.zip` intentionally omits development-only tests, rebuild notes and historical validation output. All four omitted files are preserved unchanged in `ChemBreak_Complete_Archive.zip`. This keeps the live execution package focused while retaining the exact development record.

To run regression tests, extract the **complete archive** and run from inside its `chembreak/` directory:

```bash
PYTHONPATH=src python tests/test_judge_handling.py
PYTHONPATH=src python tests/test_evaluation_quality.py
```

These run only harmless mock data and a fake managed HTTP transport. They do not load the actual Gemma model, validate real chemistry, or establish publication-quality interrater agreement. Use an independent, pre-registered human-label calibration set to measure classification accuracy, ordinal agreement and unresolved rates before a full live study.
