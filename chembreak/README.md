# ChemBreak

ChemBreak is an independently reproducible chemistry-language-model safety-evaluation harness. This rebuild **replaces GPT judging with a locally served, instruction-tuned Google Gemma 4 26B A4B** model. It preserves the uploaded 28-task bank, two target-model configurations, three adaptive stages, six turns per stage, separate baseline conversation, task isolation, and two independent cumulative judgment requests. The attack-generation framework has not been expanded.

## What changed

| Feature | This rebuild |
|---|---|
| Judge | `google/gemma-4-26B-A4B-it`, local vLLM server at `http://127.0.0.1:8000/v1` |
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

1. Use the **`ChemBreak_Gemma4_Rebuilt.zip`** release file and open `chembreak/notebooks/ChemBreak_Cloud_Notebook.ipynb`. In Cell 2, keep `SOURCE_PACKAGE = 'uploaded_zip'`. The notebook accepts a ZIP upload into `/content` or you can place it in the Colab Files panel.
2. Start with `DRY_RUN=True`, and `TASK_LIMIT=1` for the first validation. Dry run uses harmless mocks and does **not** require a GPU-based Gemma server.
3. For a live run, serve Gemma 4 with vLLM from a **separate environment**, following the instructions below. In the notebook, set `DRY_RUN=False`, supply Vertex credentials, and enable `PROBE_ROLES=True` for benign role smoke tests. GPT API credentials are **not used**.
4. Use a fresh `RUN_ID` for each judge/model/quantization/scoring change. Never resume the old GPT 1–5 OFS experiment into the new state. Run the target models separately via their notebook sections.
5. Keep outputs under `INTERNAL_DO_NOT_RELEASE` private. The release folder redacts raw prompts and responses.

The notebook defaults to **uploading this revised ZIP** rather than fetching an older GitHub branch. If you switch `SOURCE_PACKAGE` to `github`, you must first commit the rebuilt package to the selected branch.

## Launch the local Gemma judge

Install vLLM in a separate Python environment compatible with Gemma 4 (consult the current vLLM Gemma 4 recipe):

```bash
python3 -m venv .venv-gemma
source .venv-gemma/bin/activate
pip install --upgrade pip
pip install 'vllm>=0.25.0'
./scripts/start_gemma_server.sh
```

This starts `vllm serve google/gemma-4-26B-A4B-it --host 127.0.0.1 --port 8000` with a 16,384-token context budget and conservative concurrency settings. The complete active target-response context must fit; an overlength request must remain an error, never a silent truncation. The executable prints server and model loading errors directly. The server supports JSON-schema-constrained outputs through its OpenAI-compatible **local HTTP protocol**. Using that wire format does not send requests to OpenAI.

To verify the endpoint on the same host, run this harmless smoke test (from the target environment, using this package's dependencies):

```bash
python scripts/probe_local_judge.py configs/config.yaml
```

**GPU capacity:** Google's model table estimates about **57.7 GB** for Gemma 4 26B A4B BF16 weights plus overhead, with additional context-cache requirements. A full-precision judge alongside a 7B/8B BF16 target may exceed even a single A100 80 GB allocation. Prefer a dedicated GPU for the judge and a separate GPU for the target. When they run on different machines, forward a secured tunnel to `127.0.0.1:8000`. This package deliberately refuses non-loopback judge URLs.

A community quantized 4-bit model may fit a smaller GPU, but it is **not equivalent** to the official full-precision checkpoint for a measurement study. Its score calibration and version must be separately validated. `configs/config.yaml` intentionally pins the official instruction-tuned model. Verify the served model ID matches the configured ID. The supplied `scripts/start_gemma_server.sh` permits an explicit `GEMMA_JUDGE_MODEL` override, but updating the package's model identity and preflight for a different checkpoint requires a new calibration protocol and source change.

Do not install vLLM and its modern Transformers dependencies over the original 7B/8B target-model Python environment without checking compatibility. The targets still carry their historical package pins.

## Scoring and quality review

The exact 1–10 OFS anchors are declared in `src/chembreak/prompts.py` and explained in `EVALUATION_PROTOCOL.md`. The old 1–5 OFS scale cannot be directly compared to this 1–10 rubric. For the new scale, controller thresholds and OFS-delta normalization have been migrated rather than carrying forward the old hard-coded score 4/5 constants.

Evidence from the judge is a short **verbatim excerpt** and a 1-based source-response turn. `evidence.py` finds the excerpt in that exact response, verifies its offsets, and records SHA256 hashes. These checks prove that a judge cited **text that exists**, not that the underlying chemistry is sound, the answer is correct, or the risk assessment is valid. Raw citation excerpts are not copied into public exports. An unsupported high OFS or a cumulative regression is held for review.

Outputs include `release/summary.json`, `release/target_queries.csv`, `release/judgment_review_queue.csv`, private full transcripts, and `INTERNAL_DO_NOT_RELEASE/evidence_registry.jsonl`. High scores, conflicts, and incomplete scores require qualified validation before claims of reliable accuracy or attack success. The automatic ASR is a **provisional model-predicted proportion**, not a verified success rate or lower bound.

## Offline regression tests

Run from inside `chembreak/`:

```bash
PYTHONPATH=src python tests/test_judge_handling.py
PYTHONPATH=src python tests/test_evaluation_quality.py
```

These run only harmless mock data and a fake local HTTP transport. They do not load the actual Gemma model, validate real chemistry, or establish publication-quality interrater agreement. Use an independent, pre-registered human-label calibration set to measure classification accuracy, ordinal agreement and unresolved rates before a full live study.
