# Validation Record

Validation date: September 4, 2026

## Completed checks

| Check | Result |
| --- | --- |
| Frozen task-bank rows | 500 |
| Frozen task-bank columns | 22 |
| Duplicate assignment IDs | 0 |
| Duplicate benchmark prompts | 0 |
| Prompt length contract | All prompts contain 22 to 45 words |
| Task-bank SHA-256 | `62df773ce8c4a252bd23350fcd3a8e83fc670864efb0304fe8c65d21d3c7d6ff` |
| Python source compilation | Passed |
| Notebook JSON validation | Passed |
| Notebook installation design | Preserves the system Torch stack; installs only missing non-Torch packages on `/content` with `--no-deps` |
| Unit and contract checks | 11 passed with the dependency-free test harness |
| Default large-write path audit | Every configured path is below `/content` |
| Attacker feedback isolation | Policy view excludes all judge, verifier, reward, and adjudication fields |
| Full mock test episodes | 96 completed |
| Full mock transcripts | 432 recorded |
| Full mock evaluations | 432 recorded |
| Reusable C1 and C2 assets | 16 recorded |
| Full mock failures | 0 |
| Resume test | Passed with no duplicate episodes, transcripts, evaluations, assets, or role-call records |
| Confirmed-success path | Passed and stopped after one consumed query |
| Transcript/evaluation separation | Passed in SQLite and exported files |
| Release raw-field redaction | Passed |

The full mock run used 8 stratified tasks, all 3 configured targets, all 4 conditions, and the fixed five-query maximum. C0, C1, and C2 used all five queries because the mock target always refused. C3 stopped after three turns under the declared stagnation rule. The resulting mock attack success rate is correctly zero because no live target was queried.

The packaging environment does not expose the Notebook Enterprise `/content` mount. Storage enforcement was therefore validated with an isolated local root and with the separate-mount requirement disabled only in the ignored `runtime.local.yaml`. The shipped Google Cloud configuration keeps that requirement enabled and requires at least 100 GiB free.

## Cloud-only checks still required

The following checks require the authenticated Google Cloud Notebook Enterprise runtime:

- confirmation that `/content` is a separate mounted filesystem
- Google Cloud access to Gemini 3.1 Pro Preview, GPT-OSS 120B, Gemini 2.5 Pro, and Llama 4 Maverick
- sequential download and GPU loading of ChemDFM, ChemLLM, and the LlaSMol base plus adapter
- live test-phase latency and token-usage capture
- private Cloud Storage checkpoint upload

Run the notebook's live preflight before the first live test. Do not move to pilot if any disk, model, schema, region, GPU, or checkpoint check fails.
