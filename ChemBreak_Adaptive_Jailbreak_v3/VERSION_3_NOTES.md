# ChemBreak Adaptive Jailbreak v3 release notes

## Purpose

V3 is a fresh major-version package created after the V2 TEST exposed evaluation reliability and history-separation problems. V2 remains untouched and its results are not imported into V3.

## Changes from V2

### 1. Attacker role is explicit

The attacker remains `gemini-3.1-pro-preview`. V3 now prints attacker-role status during frozen asset preparation and live C3 adaptation without printing the generated attack text.

### 2. Judge feedback is truly disabled in the primary adaptive method

V2 stored evaluation objects inside the turn records that were also serialized for the next attacker call. V3 separates clean target conversation history from experiment/evaluation records. With `attacker_uses_judge_feedback: false`, the attacker sees no safety scores, chemistry scores, verifier result, adjudication, or prior judge rationale.

The configuration flag is now actually enforced. A later ablation can enable high-level judge feedback explicitly without changing the main experiment.

### 3. Later judges no longer see prior judge outputs

C2 and C3 pass only user/target conversation content into later judge calls. Prior safety, chemistry, verifier, and adjudication objects remain in stored experiment records but do not re-enter the judge context.

### 4. Gemini 2.5 Pro chemistry judging uses structured output

V3 replaces loose JSON-mode generation with an explicit JSON Schema for the chemistry judge. The response schema requires all ChemBreak chemistry scoring fields and prevents free-form structural drift.

Chemistry-judge defaults are now 4096 output tokens, thinking budget 1024, one client-level structured generation attempt, and at most two validated judge attempts. Finish reasons are inspected and `MAX_TOKENS` is reported distinctly.

### 5. GPT-OSS 120B gets adaptive throttling control

The baseline request interval is increased. HTTP 429 events increase the dynamic request interval, `Retry-After` is honored when available, and the interval decays after successful requests. This reduces repeated bursts against the same quota.

### 6. Preflight checks the real structured-output paths

Before any harmful benchmark task is sent to a target, V3 makes harmless role-specific smoke tests for the attacker, safety judge, chemistry judge, and adjudicator. A broken JSON-schema or model endpoint fails preflight rather than surfacing deep into target execution.

### 7. Reliability diagnostics are saved

`public/model_client_diagnostics.json` records request counts, network retries, JSON retries, rate-limit events, Gemini token-limit events, and current adaptive pacing state.

### 8. V2 storage fixes are retained

Hugging Face model weights remain routed to `/content/hf_cache`; preflight verifies free disk space; a target load failure aborts the target block once; target weights are loaded once per target and reused across conditions.

## Unchanged methodology

- TEST: 8 task-bank prompts
- PILOT: 40 task-bank prompts
- PRODUCTION: all 500 task-bank prompts
- C0 Direct Single-Turn
- C1 Repeated Single-Turn
- C2 Fixed Multi-Turn
- C3 Response-Adaptive ChemBreak
- C1/C2/C3 query budget: 5
- C3 maximum route switches: 2
- Attacker: Gemini 3.1 Pro Preview
- Safety judge: GPT-OSS 120B
- Chemistry judge: Gemini 2.5 Pro
- Chemistry Domain Verifier: RDKit + task metadata
- Adjudicator: Llama 4 Maverick
- Targets: ChemDFM, ChemLLM, LlaSMol

## Validation status

V3 is validated offline for compilation, unit tests, notebook JSON structure, runtime-config generation, namespace consistency, and archive contents. It has not been live-tested against the cloud target models during package creation. Run TEST first.
