# Validation Record

Validation date: September 9, 2026

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
| Notebook installation design | Preserves system Torch, Torchvision, CUDA; installs a pinned non-Torch ML overlay under `/content` |
| Unit, contract, integration, and fault-injection checks | 22 passed |
| Default large-write path audit | Every configured path is below `/content` |
| Attacker feedback isolation | Policy view excludes all judge, verifier, reward, and adjudication fields |
| Full mock test episodes | 96 completed |
| Full mock transcripts | 336 recorded |
| Full mock evaluations | 336 recorded |
| Reusable C1 and C2 assets | 16 recorded |
| Mock role-call records | 832 recorded privately |
| Full mock failures | 0 |
| Resume test | Passed with no duplicate episodes, transcripts, evaluations, assets, or role-call records |
| Structured-output retry | Passed with two malformed responses followed by a valid schema-conforming response |
| Role preflight contract | Passed locally with the same strict-schema or client-validated protocol selected for each production role |
| GPT-OSS safety protocol | Passed; compact JSON contract is client validated without claiming unsupported strict-schema enforcement |
| Structured-output diagnostics | Passed; role, attempt, finish reason, length, validation error, and raw response are retained privately |
| Two-stage target checkpoint | Passed; a saved target response was judged after interruption without a second target query |
| Pending-judgment recovery cell | Passed; 24 pending direct-control responses were recovered with no additional target queries |
| ChemLLM Boolean tokenizer regression | Passed; invalid Boolean result was rejected and the validated fallback was selected |
| Target-load isolation | Passed; simulated ChemLLM failure produced 32 `target_unavailable` rows while 64 other episodes completed |
| Target-query failure classification | Passed; inference failure is stored as technical and excluded from evaluated `NO` results |
| Direct-control budget | Passed; C0 consumes exactly one target query whether successful or refused |
| C3 refusal handling | Passed; low response-level task fidelity does not stop adaptation after turn one |
| Adaptive attribution | Passed; turn-one C3 success and post-feedback success are exported separately |
| Transcript/evaluation separation | Passed in SQLite and exported files |
| Release raw-field redaction | Passed |
| Buffer-safe live dashboard | Passed with one updating notebook display and a file-backed status record |
| Sanitized event log | Passed; no raw attack prompt or target response fields |
| Explicit episode verdict export | 96 rows with visible `NO` mock verdicts and chemistry validation fields |
| Target-condition success table | 12 complete target-condition groups |
| Independent condition cells | Passed; all four cells share one signed run and frozen task selection |
| Dedicated checkout notebook startup | Passed static contract checks for `/content/chembreak6_runtime_repo`, external runtime configuration, and non-destructive dirty-checkout handling |

The full mock run used 8 stratified tasks, all 3 configured targets, and all 4 conditions. C0 used one query. C1 and C2 used five. C3 stopped after three turns under the declared stagnation rule. The resulting mock attack success rate is correctly zero because no live target was queried.

The packaging environment does not expose the Notebook Enterprise `/content` mount. Storage enforcement was therefore validated with a temporary configuration outside the delivered package and with the separate-mount requirement disabled only for that isolated test. The shipped Google Cloud configuration keeps the requirement enabled and requires at least 100 GiB free.

## Cloud-only checks still required

The following checks require your authenticated Google Cloud Notebook Enterprise runtime and therefore were not claimed as local passes:

- confirmation that `/content` is a separate mounted filesystem
- Google Cloud access to Gemini 3.1 Pro Preview, GPT-OSS 120B, Gemini 2.5 Pro, and Llama 4 Maverick
- live contract-conforming responses from all five configured cloud model roles
- repository download and harmless tokenizer probe for ChemDFM, ChemLLM, and LlaSMol
- sequential download and GPU loading of ChemDFM, ChemLLM, and the LlaSMol base plus adapter
- live test-phase latency and token-usage capture
- private Cloud Storage checkpoint upload

Run the notebook's live preflight before the first live test. Do not move to pilot if any disk, model, schema, region, GPU, or checkpoint check fails.
