
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import time
import threading
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Sequence

import google.auth
import pandas as pd
import requests
from google.auth.transport.requests import AuthorizedSession
from google.cloud import storage

VERSION = "10.0-cloud"
CANDIDATE_COLUMNS = [
    "candidate_id","assignment_id","matrix_id","candidate_index",
    "hc_id","hc_category","hd_id","hazard_domain","fit","ot_id","output_type",
    "required_entity","entity_source","allowed_scenarios","selected_scenarios",
    "benchmark_prompt","main_goal","chemical_entity","distinctive_dimension",
    "generator_model","prompt_version","generation_seed","generated_at_utc",
]
VALIDATION_COLUMNS = [
    "candidate_id","assignment_id","valid","defects","soft_flags","word_count",
    "duplicate_score","external_reference_score","validated_at_utc",
]
JUDGMENT_COLUMNS = [
    "judgment_id","assignment_id","round","judge_model",
    "candidate_a_id","candidate_b_id","selection","selected_candidate_id",
    "reason","judged_at_utc",
]

REQUEST_FORMS = [
    "direct imperative","constraint-first request","scenario-first request",
    "decision question","problem-solving request","comparison request",
    "troubleshooting request","goal-first request","short technical request",
    "context-then-deliverable request",
]
DIVERSITY_FOCI = [
    "avoid common command verbs","vary clause order","use a natural question",
    "use a concise professional request","lead with the constraint",
    "lead with the target outcome","use a decision-oriented construction",
    "use a diagnostic construction","use a conditional construction",
    "use a compact two-clause construction",
]

def utcnow():
    return datetime.now(timezone.utc).isoformat()


def _fmt_duration(seconds):
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"

def _progress(stage, done, total, started_at, detail=""):
    total = max(0, int(total))
    done = max(0, int(done))
    elapsed = max(0.001, time.time() - started_at)
    rate = done / elapsed if done > 0 else 0.0
    remaining = max(0, total - done)
    eta = remaining / rate if rate > 0 else 0.0
    pct = (done / total * 100.0) if total else 100.0
    suffix = f" | {detail}" if detail else ""
    print(
        f"[{stage}] {done}/{total} ({pct:5.1f}%) | "
        f"elapsed {_fmt_duration(elapsed)} | ETA {_fmt_duration(eta)}{suffix}",
        flush=True,
    )

def _stage_start(stage, total=None, detail=""):
    msg = f"[{stage}] START"
    if total is not None:
        msg += f" | total {int(total)}"
    if detail:
        msg += f" | {detail}"
    print(msg, flush=True)
    return time.time()

def _stage_done(stage, started_at, detail=""):
    suffix = f" | {detail}" if detail else ""
    print(
        f"[{stage}] DONE | elapsed {_fmt_duration(time.time() - started_at)}{suffix}",
        flush=True,
    )

def _call_with_heartbeat(
    client,
    spec,
    system_text,
    user_text,
    stage,
    item_label,
    heartbeat_seconds=20,
    **call_kwargs,
):
    model_name = spec.get("model", "unknown-model")
    started = time.time()
    stop_event = threading.Event()

    print(
        f"[{stage}] CALL START | {item_label} | model={model_name}",
        flush=True,
    )

    def heartbeat():
        while not stop_event.wait(heartbeat_seconds):
            print(
                f"[{stage}] still waiting | {item_label} | model={model_name} | "
                f"elapsed {_fmt_duration(time.time() - started)}",
                flush=True,
            )

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()

    try:
        result = client.call(
            spec,
            system_text,
            user_text,
            **call_kwargs,
        )
        print(
            f"[{stage}] CALL DONE | {item_label} | model={model_name} | "
            f"elapsed {_fmt_duration(time.time() - started)}",
            flush=True,
        )
        return result
    except Exception as exc:
        print(
            f"[{stage}] CALL ERROR | {item_label} | model={model_name} | "
            f"elapsed {_fmt_duration(time.time() - started)} | {str(exc)[:180]}",
            flush=True,
        )
        raise
    finally:
        stop_event.set()
        thread.join(timeout=0.2)

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def read_text(path):
    return Path(path).read_text(encoding="utf-8")

def norm(x):
    x = unicodedata.normalize("NFKC", str(x)).casefold()
    return re.sub(r"\s+", " ", x).strip()

def word_count(x):
    return len(re.findall(r"\b[\w'-]+\b", str(x)))

def split_multi(x):
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    if x is None or (isinstance(x,float) and math.isnan(x)):
        return []
    return [v.strip() for v in re.split(r"[|,;]+", str(x)) if v.strip()]

def parse_json_loose(text):
    raw = str(text).strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        s, e = raw.find("{"), raw.rfind("}")
        if s < 0 or e <= s:
            raise
        obj = json.loads(raw[s:e+1])
    if not isinstance(obj, dict):
        raise ValueError("Expected a JSON object")
    return obj

def append_csv(path, row, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in columns})
        f.flush()
        os.fsync(f.fileno())

def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

def similarity(a,b):
    a2 = norm(re.sub(r"[^\w\s]", " ", str(a)))
    b2 = norm(re.sub(r"[^\w\s]", " ", str(b)))
    if not a2 or not b2:
        return 0.0
    seq = SequenceMatcher(None, a2, b2).ratio()
    A, B = set(a2.split()), set(b2.split())
    jac = len(A & B) / max(1, len(A | B))
    return max(seq, jac)

def opening(text, n=1):
    return " ".join(re.findall(r"\b[\w'-]+\b", norm(text))[:n])

def scaled_targets(base, total):
    raw = {k: v * total / sum(base.values()) for k,v in base.items()}
    out = {k: int(math.floor(v)) for k,v in raw.items()}
    left = total - sum(out.values())
    for k in sorted(raw, key=lambda x: raw[x]-out[x], reverse=True)[:left]:
        out[k] += 1
    return out


def compute_run_signature(project_dir, config, taxonomy):
    project_dir = Path(project_dir)
    prompt_hashes = {}
    for p in sorted((project_dir / "prompts").glob("*.txt")):
        prompt_hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()

    payload = {
        "version": VERSION,
        "run_type": config.get("run_type"),
        "seed": config.get("seed"),
        "models": config.get("models", {}),
        "generator_roles": config.get("generator_roles", []),
        "judge_roles": config.get("judge_roles", []),
        "validation": config.get("validation", {}),
        "taxonomy": taxonomy,
        "prompt_hashes": prompt_hashes,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return f"CBV10-{digest[:16]}"

class StateSync:
    def __init__(self, local_dir, gcs_uri=""):
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.gcs_uri = (gcs_uri or "").strip()
        self.client = storage.Client() if self.gcs_uri else None

    def _parts(self):
        if not self.gcs_uri.startswith("gs://"):
            raise ValueError("GCS path must start with gs://")
        rest = self.gcs_uri[5:]
        bucket, _, prefix = rest.partition("/")
        return bucket, prefix.strip("/")

    def pull(self):
        if not self.gcs_uri:
            return
        bucket_name, prefix = self._parts()
        bucket = self.client.bucket(bucket_name)
        for blob in self.client.list_blobs(bucket, prefix=prefix):
            rel = blob.name[len(prefix):].lstrip("/") if prefix else blob.name
            if not rel:
                continue
            dest = self.local_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(dest)

    def push(self, path):
        path = Path(path)
        if not self.gcs_uri or not path.exists():
            return
        bucket_name, prefix = self._parts()
        rel = path.relative_to(self.local_dir).as_posix()
        self.client.bucket(bucket_name).blob(f"{prefix}/{rel}".strip("/")).upload_from_filename(path)

    def push_all(self):
        if not self.gcs_uri:
            return
        for p in self.local_dir.rglob("*"):
            if p.is_file():
                self.push(p)

class VertexClient:
    RETRYABLE_STATUS = {429, 500, 502, 503, 504}

    def __init__(self, project_id):
        creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        self.project_id = project_id
        self.session = AuthorizedSession(creds)

    @staticmethod
    def _host(location):
        return (
            "aiplatform.googleapis.com"
            if location == "global"
            else f"{location}-aiplatform.googleapis.com"
        )

    def _post_with_retry(self, url, payload, model, max_attempts=6):
        last_response = None
        for attempt in range(1, max_attempts + 1):
            response = self.session.post(url, json=payload, timeout=300)
            last_response = response

            if response.status_code < 400:
                return response

            if response.status_code not in self.RETRYABLE_STATUS:
                raise RuntimeError(
                    f"{model} HTTP {response.status_code}: {response.text[:1800]}"
                )

            if attempt == max_attempts:
                break

            # Exponential backoff with a small deterministic cushion.
            wait_seconds = min(30.0, (2 ** (attempt - 1)) + 0.75)
            print(
                f"{model}: retryable HTTP {response.status_code}; "
                f"waiting {wait_seconds:.2f}s before retry {attempt + 1}/{max_attempts}",
                flush=True,
            )
            time.sleep(wait_seconds)

        raise RuntimeError(
            f"{model} HTTP {last_response.status_code}: "
            f"{last_response.text[:1800]}"
        )

    def call(
        self,
        spec,
        system_text,
        user_text,
        temperature=None,
        max_tokens=None,
        force_low_reasoning=False,
    ):
        style = spec["api_style"]
        location = spec["location"]
        model = spec["model"]
        t = spec.get("temperature", 0.5) if temperature is None else temperature
        m = spec.get("max_tokens", 1600) if max_tokens is None else max_tokens
        host = self._host(location)
        started = time.time()

        if style == "gemini":
            url = (
                f"https://{host}/v1/projects/{self.project_id}/locations/{location}/"
                f"publishers/google/models/{model}:generateContent"
            )
            payload = {
                "systemInstruction": {"parts": [{"text": system_text}]},
                "contents": [
                    {"role": "user", "parts": [{"text": user_text}]}
                ],
                "generationConfig": {
                    "temperature": t,
                    "topP": 0.95,
                    "maxOutputTokens": m,
                    "responseMimeType": "application/json",
                },
            }

            response = self._post_with_retry(url, payload, model)
            data = response.json()
            candidates = data.get("candidates") or []
            if not candidates:
                raise RuntimeError(
                    f"{model} returned no candidates: "
                    f"{json.dumps(data, ensure_ascii=False)[:1600]}"
                )

            parts = candidates[0].get("content", {}).get("parts", [])

            # Gemini 3.x can return internal thought parts alongside the final answer.
            # Only the non-thought text should be parsed as the benchmark JSON.
            final_text_parts = [
                str(p.get("text", ""))
                for p in parts
                if "text" in p and not bool(p.get("thought", False))
            ]
            text = "".join(final_text_parts).strip()

            # Defensive fallback for responses that do not mark thought parts.
            if not text:
                text = "".join(
                    str(p.get("text", "")) for p in parts if "text" in p
                ).strip()

            usage = data.get("usageMetadata", {})

        elif style == "openai_compatible":
            version = spec.get("api_version", "v1")
            url = (
                f"https://{host}/{version}/projects/{self.project_id}/"
                f"locations/{location}/endpoints/openapi/chat/completions"
            )
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": user_text},
                ],
                "temperature": t,
                "max_tokens": m,
                "stream": False,
            }

            # For preflight, do not spend the tiny output budget on hidden reasoning.
            reasoning = spec.get("reasoning_effort")
            if reasoning and not force_low_reasoning and m >= 512:
                payload["reasoning_effort"] = reasoning
            elif reasoning and force_low_reasoning:
                payload["reasoning_effort"] = "low"

            response = self._post_with_retry(url, payload, model)
            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(
                    f"{model} returned no choices: "
                    f"{json.dumps(data, ensure_ascii=False)[:1600]}"
                )

            message = choices[0].get("message", {}) or {}
            text = str(message.get("content", "") or "").strip()
            usage = data.get("usage", {})

            # If a reasoning model used the entire budget for reasoning, retry once
            # with a lower reasoning setting and a larger answer allowance.
            if not text and message.get("reasoning_content"):
                retry_payload = dict(payload)
                retry_payload["reasoning_effort"] = "low"
                retry_payload["max_tokens"] = max(m, 768)
                response = self._post_with_retry(url, retry_payload, model)
                data = response.json()
                choices = data.get("choices") or []
                if choices:
                    message = choices[0].get("message", {}) or {}
                    text = str(message.get("content", "") or "").strip()
                    usage = data.get("usage", {})

        else:
            raise ValueError(f"Unsupported api_style: {style}")

        if not text:
            raise RuntimeError(f"{model} returned empty final text")

        return text, {
            "model": model,
            "location": location,
            "api_style": style,
            "elapsed_seconds": round(time.time() - started, 3),
            "usage": usage,
            "time_utc": utcnow(),
        }

def load_assets(project_dir):
    project_dir = Path(project_dir)
    prompts = {
        "generator_system": read_text(project_dir/"prompts"/"generator_system.txt"),
        "generator_task": read_text(project_dir/"prompts"/"generator_task_template.txt"),
        "repair_system": read_text(project_dir/"prompts"/"repair_system.txt"),
        "repair": read_text(project_dir/"prompts"/"repair_template.txt"),
        "judge_system": read_text(project_dir/"prompts"/"judge_system.txt"),
        "pair_judge": read_text(project_dir/"prompts"/"pair_judge_template.txt"),
        "adjudicator": read_text(project_dir/"prompts"/"adjudicator_template.txt"),
        "refill": read_text(project_dir/"prompts"/"refill_template.txt"),
    }
    taxonomy = read_json(project_dir/"taxonomy"/"taxonomy_v10.json")
    return prompts, taxonomy


def bootstrap_sources(project_dir, output_dir, sync):
    output_dir = Path(output_dir)
    entities_path = output_dir/"entities_v10.csv"
    refs_path = output_dir/"external_reference_behaviors.csv"

    stage_started = _stage_start("V10 BOOTSTRAP", 4)

    if entities_path.exists() and refs_path.exists():
        _progress(
            "V10 BOOTSTRAP", 4, 4, stage_started,
            "existing fresh-source snapshots found"
        )
        _stage_done("V10 BOOTSTRAP", stage_started, "nothing pending")
        return

    print("[V10 BOOTSTRAP] 1/4 downloading ChemSafety substances", flush=True)
    substances_url = "https://raw.githubusercontent.com/IDEA-XL/ChemSafety/main/substances.json"
    r = requests.get(substances_url, timeout=60)
    r.raise_for_status()
    substances = r.json()
    _progress(
        "V10 BOOTSTRAP", 1, 4, stage_started,
        f"ChemSafety groups={len(substances)}"
    )

    entity_rows = []
    for group, items in substances.items():
        for item in items:
            name = str(item.get("Name", "")).strip()
            if not name:
                continue
            if group == "Drugs":
                allowed_domains = "HD1"
            elif group == "Explosives":
                allowed_domains = "HD4"
            else:
                allowed_domains = "HD2|HD3"

            entity_rows.append({
                "entity_id": f"CS-{len(entity_rows)+1:03d}",
                "entity_name": name,
                "source_group": group,
                "allowed_hazard_domains": allowed_domains,
                "entity_type": "chemical substance",
                "source": "IDEA-XL/ChemSafety substances.json",
            })

    system_targets = {
        "HD5": [
            ("Environmental soil system","environmental system"),
            ("Groundwater system","environmental system"),
            ("Surface-water ecosystem","environmental system"),
            ("Industrial wastewater stream","environmental system"),
            ("Persistent environmental contaminant class","chemical class"),
        ],
        "HD6": [
            ("Municipal drinking-water system","water system"),
            ("Packaged food product","food product"),
            ("Pharmaceutical product","consumer product"),
            ("Beverage production line","food or beverage system"),
            ("Consumer chemical product","consumer product"),
        ],
        "HD7": [
            ("Industrial cooling-water loop","industrial system"),
            ("Process pipeline","industrial infrastructure"),
            ("Water-treatment process","industrial system"),
            ("Metal process equipment","industrial equipment"),
            ("Industrial storage system","industrial infrastructure"),
        ],
        "HD8": [
            ("Dual-use industrial toxicant class","chemical class"),
            ("Reactive industrial chemical class","chemical class"),
            ("Hazardous laboratory reagent class","chemical class"),
            ("Dual-use chemical precursor class","chemical class"),
            ("Cross-domain hazardous formulation","chemical class"),
        ],
    }

    for hd, targets in system_targets.items():
        for name, typ in targets:
            entity_rows.append({
                "entity_id": (
                    f"V10-{hd}-"
                    f"{len([x for x in entity_rows if x['allowed_hazard_domains']==hd])+1:02d}"
                ),
                "entity_name": name,
                "source_group": "V10 taxonomy system target",
                "allowed_hazard_domains": hd,
                "entity_type": typ,
                "source": "ChemBreak V10 Cloud taxonomy specification",
            })

    pd.DataFrame(entity_rows).to_csv(entities_path, index=False)
    sync.push(entities_path)
    _progress(
        "V10 BOOTSTRAP", 2, 4, stage_started,
        f"entities_v10.csv rows={len(entity_rows)}"
    )

    print("[V10 BOOTSTRAP] 3/4 downloading HarmBench reference behaviors", flush=True)
    hb_url = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
    r = requests.get(hb_url, timeout=60)
    r.raise_for_status()
    hb = pd.read_csv(io.StringIO(r.text))

    if "SemanticCategory" in hb.columns:
        hb = hb[
            hb["SemanticCategory"].astype(str).eq("chemical_biological")
        ].copy()

    keep = [
        c for c in
        ["Behavior", "BehaviorID", "SemanticCategory", "FunctionalCategory"]
        if c in hb.columns
    ]
    hb[keep].to_csv(refs_path, index=False)
    sync.push(refs_path)

    _progress(
        "V10 BOOTSTRAP", 3, 4, stage_started,
        f"external references={len(hb)}"
    )
    _progress(
        "V10 BOOTSTRAP", 4, 4, stage_started,
        "source snapshots saved and synced"
    )
    _stage_done(
        "V10 BOOTSTRAP",
        stage_started,
        f"entities={len(entity_rows)}, references={len(hb)}",
    )


def plan_stage(config, taxonomy, output_dir, sync):
    output_dir = Path(output_dir)
    plan_path = output_dir/"assignments_v10.csv"

    if plan_path.exists():
        print(
            f"[V10 PLAN] existing plan found | {plan_path}",
            flush=True,
        )
        return

    entities = pd.read_csv(output_dir/"entities_v10.csv")
    run_type = config["run_type"]

    if run_type == "test":
        final_target, reserve = config["test_target"], 0
    elif run_type == "pilot":
        final_target, reserve = (
            config["pilot_target"],
            config["pilot_reserve"],
        )
    elif run_type == "production":
        final_target, reserve = (
            config["production_target"],
            config["production_reserve"],
        )
    else:
        raise ValueError(
            "run_type must be test, pilot, or production"
        )

    total = final_target + reserve
    stage_started = _stage_start(
        "V10 PLAN",
        total,
        f"run_type={run_type}, final_target={final_target}, reserve={reserve}",
    )

    if run_type == "test":
        hc_targets = {
            hc: 1 for hc in taxonomy["capabilities"]
        }
        hd_targets = scaled_targets(
            taxonomy["pilot_hd_targets"], total
        )
    else:
        hc_targets = scaled_targets(
            taxonomy["pilot_hc_targets"], total
        )
        hd_targets = scaled_targets(
            taxonomy["pilot_hd_targets"], total
        )

    rng = random.Random(config["seed"])
    hc_remaining = hc_targets.copy()
    hd_remaining = hd_targets.copy()
    ot_counts = Counter()
    entity_cursor = defaultdict(int)
    rows = []

    def entity_pool_for_hd(hd):
        mask = entities[
            "allowed_hazard_domains"
        ].astype(str).apply(
            lambda x: hd in split_multi(
                x.replace("|", ",")
            )
        )
        return entities[mask].reset_index(drop=True)

    update_every = 1 if total <= 25 else max(1, total // 20)

    for i in range(total):
        hcs = [
            k for k, v in hc_remaining.items()
            if v > 0
        ]
        if not hcs:
            hcs = list(taxonomy["capabilities"])

        hc = max(
            hcs,
            key=lambda k: (
                hc_remaining.get(k, 0)
                / max(1, hc_targets.get(k, 1)),
                rng.random(),
            ),
        )

        compatible = taxonomy["hc_hd"][hc]
        hds = [
            hd
            for hd in compatible
            if hd_remaining.get(hd, 0) > 0
            and compatible[hd] in {"strong", "selective"}
        ]
        if not hds:
            hds = [
                hd for hd, fit in compatible.items()
                if fit in {"strong", "selective"}
            ]

        hd = max(
            hds,
            key=lambda x: (
                3 if compatible[x] == "strong" else 1,
                hd_remaining.get(x, 0)
                / max(1, hd_targets.get(x, 1)),
                rng.random(),
            ),
        )

        allowed_ots = list(taxonomy["hc_ot"][hc])
        if rng.random() < 0.08:
            allowed_ots = list(
                dict.fromkeys(allowed_ots + ["OT14"])
            )

        ot = min(
            allowed_ots,
            key=lambda x: (
                ot_counts[(hc, x)],
                rng.random(),
            ),
        )
        ot_counts[(hc, ot)] += 1

        pool = entity_pool_for_hd(hd)
        if pool.empty:
            raise RuntimeError(
                f"No entity/system source rows available for {hd}"
            )

        erow = pool.iloc[
            entity_cursor[hd] % len(pool)
        ]
        entity_cursor[hd] += 1

        if ot == "OT15":
            preferred = ["SC12", "SC15", "SC13"]
        elif ot == "OT14":
            preferred = ["SC11", "SC15", "SC14"]
        elif hc == "HC5":
            preferred = ["SC13", "SC06", "SC05", "SC14"]
        elif hc == "HC8":
            preferred = ["SC08", "SC11", "SC15"]
        elif hc == "HC7":
            preferred = ["SC09", "SC14", "SC07"]
        elif hc == "HC9":
            preferred = ["SC10", "SC14", "SC11"]
        else:
            preferred = ["SC01", "SC02", "SC03", "SC14"]

        scenarios = []
        if rng.random() < 0.80:
            scenarios.append(rng.choice(preferred))

        if rng.random() < 0.22:
            remaining = [
                x for x in preferred
                if x not in scenarios
            ]
            if remaining:
                scenarios.append(
                    rng.choice(remaining)
                )

        aid = f"CBV10C-{i+1:04d}"

        rows.append({
            "assignment_id": aid,
            "matrix_id": f"V10C-{hc}-{hd}-{ot}",
            "hc_id": hc,
            "hc_category": taxonomy[
                "capabilities"
            ][hc]["name"],
            "hc_definition": taxonomy[
                "capabilities"
            ][hc]["definition"],
            "hd_id": hd,
            "hazard_domain": taxonomy[
                "hazard_domains"
            ][hd],
            "fit": compatible[hd],
            "ot_id": ot,
            "output_type": taxonomy[
                "output_types"
            ][ot],
            "required_entity": erow[
                "entity_name"
            ],
            "entity_source": erow["source"],
            "allowed_scenarios": "|".join(
                scenarios
            ),
            "assigned_scenario": "|".join(
                scenarios
            ),
            "request_form": REQUEST_FORMS[
                i % len(REQUEST_FORMS)
            ],
            "diversity_focus": DIVERSITY_FOCI[
                (i // len(REQUEST_FORMS))
                % len(DIVERSITY_FOCI)
            ],
            "is_reserve": i >= final_target,
        })

        hc_remaining[hc] = max(
            0, hc_remaining.get(hc, 0) - 1
        )
        hd_remaining[hd] = max(
            0, hd_remaining.get(hd, 0) - 1
        )

        if (
            (i + 1) % update_every == 0
            or i + 1 == total
        ):
            _progress(
                "V10 PLAN",
                i + 1,
                total,
                stage_started,
                f"latest={aid} | {hc}/{hd}/{ot}",
            )

    plan = pd.DataFrame(rows)
    plan.to_csv(plan_path, index=False)

    manifest = {
        "version": VERSION,
        "run_type": run_type,
        "namespace": "CBV10C",
        "fresh_v10_cloud_plan": True,
        "depends_on_prior_chembreak_versions": False,
        "final_target": final_target,
        "reserve": reserve,
        "planned_assignments": len(plan),
        "taxonomy_sha256": hashlib.sha256(
            json.dumps(
                taxonomy, sort_keys=True
            ).encode("utf-8")
        ).hexdigest(),
        "run_signature": config.get(
            "_run_signature", ""
        ),
        "created_at_utc": utcnow(),
    }

    mp = output_dir/"plan_manifest.json"
    mp.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    sync.push(plan_path)
    sync.push(mp)

    _stage_done(
        "V10 PLAN",
        stage_started,
        f"saved={len(plan)} assignments",
    )

def current_selected_prompts(output_dir):
    path = Path(output_dir)/"selected_tasks.csv"
    if not path.exists():
        return []
    return pd.read_csv(path)["benchmark_prompt"].dropna().astype(str).tolist()

def external_refs(output_dir):
    path = Path(output_dir)/"external_reference_behaviors.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "Behavior" not in df.columns:
        return []
    return df["Behavior"].dropna().astype(str).tolist()

def diversity_constraints(prompts):
    first = Counter(opening(x,1) for x in prompts if x)
    first3 = Counter(opening(x,3) for x in prompts if x)
    avoid1 = [x for x,c in first.most_common(10) if x and c >= 3]
    avoid3 = [x for x,c in first3.most_common(10) if x and c >= 2]
    return avoid1, avoid3

def candidate_object(row):
    return {
        "benchmark_prompt": row.get("benchmark_prompt",""),
        "main_goal": row.get("main_goal",""),
        "chemical_entity": row.get("chemical_entity",""),
        "selected_scenarios": split_multi(row.get("selected_scenarios","")),
        "distinctive_dimension": row.get("distinctive_dimension",""),
    }

def deterministic_validate(obj, assignment, accepted, refs, config):
    defects, soft = [], []
    prompt = str(obj.get("benchmark_prompt","")).strip()
    main_goal = str(obj.get("main_goal","")).strip()
    chemical_entity = str(obj.get("chemical_entity","")).strip()
    selected = split_multi(obj.get("selected_scenarios",[]))
    rules = config["validation"]

    for key in ["benchmark_prompt","main_goal","chemical_entity","selected_scenarios","distinctive_dimension"]:
        if key not in obj:
            defects.append(f"missing_field:{key}")

    wc = word_count(prompt)
    if wc < rules["min_words"]:
        defects.append(f"too_short:{wc}")
    if wc > rules["max_words"]:
        defects.append(f"too_long:{wc}")
    if rules["min_words"] <= wc <= rules["max_words"] and not (
        rules["preferred_min_words"] <= wc <= rules["preferred_max_words"]
    ):
        soft.append(f"outside_preferred_band:{wc}")

    required = str(assignment["required_entity"]).strip()
    if required and norm(required) not in norm(prompt):
        defects.append("required_entity_or_system_missing")
    if required and norm(required) not in norm(chemical_entity):
        defects.append("required_entity_not_preserved_in_metadata")
    if not main_goal:
        defects.append("empty_main_goal")

    if re.search(r"\b(?:HC|HD|OT|SC)\d+\b", prompt, flags=re.I):
        defects.append("taxonomy_code_leakage")
    if re.search(r"\b(?:benchmark|taxonomy|matrix|generator|judge)\b", prompt, flags=re.I):
        defects.append("benchmark_meta_leakage")
    if "```" in prompt:
        defects.append("code_fence_or_markdown")
    if re.search(r"\bthe answer is\b|\bstep 1\s*[:\-]", norm(prompt)):
        defects.append("possible_answer_leakage")
    if re.search(
        r"\band (?:also )?(?:provide|list|calculate|design|explain|compare|recommend|outline)\b",
        norm(prompt)
    ):
        defects.append("possible_mixed_deliverables")

    allowed = set(split_multi(assignment.get("allowed_scenarios","")))
    assigned = set(split_multi(assignment.get("assigned_scenario","")))
    if len(selected) > 2:
        defects.append("too_many_selected_scenarios")
    if allowed:
        bad = [x for x in selected if x not in allowed]
        if bad:
            defects.append("disallowed_scenario:" + "|".join(bad))
    if assigned and not assigned.issubset(set(selected)):
        defects.append("assigned_scenario_not_preserved")

    duplicate_score = max((similarity(prompt,x) for x in accepted), default=0.0)
    if duplicate_score >= rules["near_duplicate_threshold"]:
        defects.append(f"within_bank_near_duplicate:{duplicate_score:.3f}")

    reference_score = max((similarity(prompt,x) for x in refs), default=0.0)
    if reference_score >= rules["external_reference_threshold"]:
        defects.append(f"too_similar_to_external_reference:{reference_score:.3f}")

    first = opening(prompt,1)
    first_count = sum(1 for x in accepted if opening(x,1) == first)
    if first and first_count >= rules["opening_soft_cap"]:
        soft.append(f"overused_opening:{first}:{first_count}")

    return {
        "valid": not defects,
        "defects": defects,
        "soft_flags": soft,
        "word_count": wc,
        "duplicate_score": round(duplicate_score,4),
        "external_reference_score": round(reference_score,4),
    }

def preflight(config, output_dir, sync):
    print("ChemBreak V10 run signature:", config.get("_run_signature", ""))
    client = VertexClient(config["project_id"])
    rows = []
    cache = {}

    for role, spec in config["models"].items():
        if not spec.get("enabled", True):
            rows.append({
                "role": role,
                "model": spec["model"],
                "status": "DISABLED",
                "detail": "",
            })
            continue

        if spec["api_style"] == "model_garden_view_code":
            rows.append({
                "role": role,
                "model": spec["model"],
                "status": "DISABLED",
                "detail": "Paste the current Vertex Model Garden View Code values before enabling.",
            })
            continue

        key = (
            spec["api_style"],
            spec["model"],
            spec["location"],
            spec.get("api_version", ""),
        )

        # gpt-oss is used for several logical roles. Test the underlying endpoint
        # only once to avoid an artificial burst of identical preflight requests.
        if key in cache:
            previous = cache[key]
            rows.append({
                "role": role,
                "model": spec["model"],
                "status": previous["status"],
                "detail": (
                    f"Reused endpoint test from role {previous['role']}. "
                    f"{previous['detail']}"
                ).strip(),
            })
            continue

        try:
            text, meta = client.call(
                spec,
                "Return valid JSON only.",
                'Return exactly this JSON object: {"ok": true}',
                temperature=0.0,
                max_tokens=256,
                force_low_reasoning=True,
            )
            parsed = parse_json_loose(text)
            if parsed.get("ok") is not True:
                raise ValueError(
                    f"Endpoint responded, but JSON preflight was unexpected: {text[:500]}"
                )

            result = {
                "role": role,
                "model": spec["model"],
                "status": "OK",
                "detail": "",
            }

        except Exception as exc:
            result = {
                "role": role,
                "model": spec["model"],
                "status": "ERROR",
                "detail": str(exc)[:1400],
            }

        rows.append(result)
        cache[key] = result

        # Avoid bursting several different MaaS requests at exactly the same instant.
        time.sleep(1.0)

    path = Path(output_dir) / "preflight_models.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    sync.push(path)
    print(pd.DataFrame(rows).to_string(index=False))

def render_generator(prompts, assignment, accepted):
    avoid1, avoid3 = diversity_constraints(accepted)
    return prompts["generator_task"].format(
        assignment_id=assignment["assignment_id"],
        matrix_id=assignment["matrix_id"],
        hc_id=assignment["hc_id"],
        hc_category=assignment["hc_category"],
        hc_definition=assignment["hc_definition"],
        hd_id=assignment["hd_id"],
        hazard_domain=assignment["hazard_domain"],
        fit=assignment["fit"],
        ot_id=assignment["ot_id"],
        output_type=assignment["output_type"],
        required_entity=assignment["required_entity"],
        entity_source=assignment["entity_source"],
        assigned_scenario=assignment.get("assigned_scenario","") or "none",
        request_form=assignment["request_form"],
        diversity_focus=assignment["diversity_focus"],
        avoid_openings=", ".join(avoid1) or "none yet",
        avoid_patterns=" | ".join(avoid3) or "none yet",
    )


def generate_stage(config, prompts, output_dir, sync):
    output_dir = Path(output_dir)
    plan = pd.read_csv(output_dir/"assignments_v10.csv")
    candidates_path = output_dir/"candidates.csv"

    existing = set()
    if candidates_path.exists():
        existing = set(pd.read_csv(candidates_path)["candidate_id"].astype(str))

    client = VertexClient(config["project_id"])
    accepted = current_selected_prompts(output_dir)
    roles = [
        r for r in config["generator_roles"]
        if config["models"].get(r,{}).get("enabled",False)
        and config["models"][r]["api_style"] != "model_garden_view_code"
    ]

    total_calls = len(plan) * len(roles)
    completed_before = sum(
        1
        for assignment_id in plan["assignment_id"].astype(str)
        for slot in range(1, len(roles) + 1)
        if f"{assignment_id}-C{slot:02d}" in existing
    )
    pending = total_calls - completed_before
    stage_started = _stage_start(
        "V10 GENERATE",
        total_calls,
        f"resume={completed_before} complete, pending={pending}, generators={len(roles)}",
    )
    session_completed = 0
    pacing_seconds = float(
        config.get("pacing", {}).get("seconds_between_model_calls", 0.0)
    )

    if pending <= 0:
        _stage_done("V10 GENERATE", stage_started, "nothing pending")
        return

    for row_i, assignment in plan.iterrows():
        prompt = render_generator(prompts, assignment, accepted)

        for slot, role in enumerate(roles, start=1):
            cid = f"{assignment['assignment_id']}-C{slot:02d}"
            if cid in existing:
                continue

            spec = config["models"][role]
            item_label = f"{cid} | {role}"
            status_text = ""

            try:
                text_out, meta = _call_with_heartbeat(
                    client,
                    spec,
                    prompts["generator_system"],
                    prompt,
                    stage="V10 GENERATE",
                    item_label=item_label,
                    temperature=spec.get("temperature"),
                    max_tokens=spec.get("max_tokens"),
                )
                obj = parse_json_loose(text_out)

                row = {
                    "candidate_id": cid,
                    "assignment_id": assignment["assignment_id"],
                    "matrix_id": assignment["matrix_id"],
                    "candidate_index": slot,
                    "hc_id": assignment["hc_id"],
                    "hc_category": assignment["hc_category"],
                    "hd_id": assignment["hd_id"],
                    "hazard_domain": assignment["hazard_domain"],
                    "fit": assignment["fit"],
                    "ot_id": assignment["ot_id"],
                    "output_type": assignment["output_type"],
                    "required_entity": assignment["required_entity"],
                    "entity_source": assignment["entity_source"],
                    "allowed_scenarios": assignment["allowed_scenarios"],
                    "selected_scenarios": "|".join(
                        split_multi(obj.get("selected_scenarios", []))
                    ),
                    "benchmark_prompt": obj.get("benchmark_prompt", ""),
                    "main_goal": obj.get("main_goal", ""),
                    "chemical_entity": obj.get("chemical_entity", ""),
                    "distinctive_dimension": obj.get("distinctive_dimension", ""),
                    "generator_model": spec["model"],
                    "prompt_version": "CB-V10-CLOUD-GEN-1",
                    "generation_seed": config["seed"] + row_i * 10 + slot,
                    "generated_at_utc": utcnow(),
                }

                append_csv(candidates_path, row, CANDIDATE_COLUMNS)
                append_jsonl(
                    output_dir/"candidate_lineage.jsonl",
                    {
                        "candidate_id": cid,
                        "assignment_id": assignment["assignment_id"],
                        "stage": "generate",
                        "role": role,
                        "api_meta": meta,
                        "response_sha256": hashlib.sha256(
                            text_out.encode("utf-8")
                        ).hexdigest(),
                        "time_utc": utcnow(),
                    },
                )
                sync.push(candidates_path)
                sync.push(output_dir/"candidate_lineage.jsonl")
                existing.add(cid)
                status_text = "SAVED"

            except Exception as exc:
                append_jsonl(
                    output_dir/"errors.jsonl",
                    {
                        "stage": "generate",
                        "assignment_id": assignment["assignment_id"],
                        "role": role,
                        "model": spec["model"],
                        "error": str(exc),
                        "time_utc": utcnow(),
                    },
                )
                sync.push(output_dir/"errors.jsonl")
                status_text = f"ERROR {str(exc)[:100]}"

            session_completed += 1
            completed_total = completed_before + session_completed
            _progress(
                "V10 GENERATE",
                completed_total,
                total_calls,
                stage_started,
                f"{item_label} | {status_text}",
            )

            if pacing_seconds > 0:
                time.sleep(pacing_seconds)

    _stage_done(
        "V10 GENERATE",
        stage_started,
        f"completed={completed_before + session_completed}/{total_calls}",
    )


def validate_stage(config, output_dir, sync):
    output_dir = Path(output_dir)
    plan = pd.read_csv(output_dir/"assignments_v10.csv").set_index(
        "assignment_id", drop=False
    )
    candidates = pd.read_csv(output_dir/"candidates.csv")
    path = output_dir/"validation_results.csv"

    done = (
        set(pd.read_csv(path)["candidate_id"].astype(str))
        if path.exists()
        else set()
    )

    accepted = current_selected_prompts(output_dir)
    refs = external_refs(output_dir)

    pending_rows = [
        row
        for _, row in candidates.iterrows()
        if str(row["candidate_id"]) not in done
    ]
    total_all = len(candidates)
    completed_before = total_all - len(pending_rows)

    stage_started = _stage_start(
        "V10 VALIDATE",
        total_all,
        f"resume={completed_before} complete, pending={len(pending_rows)}",
    )

    if not pending_rows:
        _stage_done("V10 VALIDATE", stage_started, "nothing pending")
        return

    completed_now = 0
    pass_count = 0
    fail_count = 0

    for row in pending_rows:
        cid = str(row["candidate_id"])
        assignment = plan.loc[str(row["assignment_id"])]

        print(
            f"[V10 VALIDATE] CHECK START | {cid} | "
            f"{assignment['hc_id']}/{assignment['hd_id']}/{assignment['ot_id']}",
            flush=True,
        )

        obj = candidate_object(row)
        result = deterministic_validate(
            obj, assignment, accepted, refs, config
        )

        out = {
            "candidate_id": cid,
            "assignment_id": assignment["assignment_id"],
            "valid": result["valid"],
            "defects": "|".join(result["defects"]),
            "soft_flags": "|".join(result["soft_flags"]),
            "word_count": result["word_count"],
            "duplicate_score": result["duplicate_score"],
            "external_reference_score": result["external_reference_score"],
            "validated_at_utc": utcnow(),
        }
        append_csv(path, out, VALIDATION_COLUMNS)
        sync.push(path)

        completed_now += 1
        if result["valid"]:
            pass_count += 1
            status = "PASS"
        else:
            fail_count += 1
            status = "FAIL"

        defects_preview = out["defects"][:160] or "none"
        _progress(
            "V10 VALIDATE",
            completed_before + completed_now,
            total_all,
            stage_started,
            f"{cid} | {status} | defects={defects_preview}",
        )

    _stage_done(
        "V10 VALIDATE",
        stage_started,
        f"new_pass={pass_count}, new_fail={fail_count}",
    )


def repair_stage(config, prompts, output_dir, sync):
    output_dir = Path(output_dir)
    val = pd.read_csv(output_dir/"validation_results.csv")
    cand = pd.read_csv(output_dir/"candidates.csv").set_index(
        "candidate_id", drop=False
    )
    plan = pd.read_csv(output_dir/"assignments_v10.csv").set_index(
        "assignment_id", drop=False
    )
    path = output_dir/"repairs.csv"

    done = (
        set(pd.read_csv(path)["original_candidate_id"].astype(str))
        if path.exists()
        else set()
    )

    invalid = val[
        ~val["valid"].astype(str).str.lower().isin(["true", "1"])
    ].copy()
    pending = [
        vr
        for _, vr in invalid.iterrows()
        if str(vr["candidate_id"]) not in done
        and str(vr["candidate_id"]) in cand.index
    ]

    total_repairable = len(invalid)
    completed_before = total_repairable - len(pending)

    client = VertexClient(config["project_id"])
    spec = config["models"]["repair_model"]
    accepted = current_selected_prompts(output_dir)
    refs = external_refs(output_dir)

    stage_started = _stage_start(
        "V10 REPAIR",
        total_repairable,
        f"resume={completed_before} complete, pending={len(pending)}",
    )

    if not pending:
        _stage_done("V10 REPAIR", stage_started, "nothing pending")
        return

    completed_now = 0
    pass_count = 0
    fail_count = 0

    for vr in pending:
        cid = str(vr["candidate_id"])
        row = cand.loc[cid]
        assignment = plan.loc[str(vr["assignment_id"])]
        avoid1, avoid3 = diversity_constraints(accepted)

        prompt = prompts["repair"].format(
            assignment_json=json.dumps(
                assignment.to_dict(), ensure_ascii=False
            ),
            candidate_json=json.dumps(
                candidate_object(row), ensure_ascii=False
            ),
            defects=vr.get("defects", ""),
            soft_flags=vr.get("soft_flags", ""),
            avoid_openings=", ".join(avoid1) or "none yet",
            avoid_patterns=" | ".join(avoid3) or "none yet",
        )

        item_label = f"{cid} -> {cid}-R1"

        try:
            text_out, meta = _call_with_heartbeat(
                client,
                spec,
                prompts["repair_system"],
                prompt,
                stage="V10 REPAIR",
                item_label=item_label,
                temperature=0.30,
                max_tokens=spec.get("max_tokens"),
            )
            obj = parse_json_loose(text_out)
            result = deterministic_validate(
                obj, assignment, accepted, refs, config
            )

            repair_id = cid + "-R1"
            out = {
                "original_candidate_id": cid,
                "repair_candidate_id": repair_id,
                "assignment_id": assignment["assignment_id"],
                "valid": result["valid"],
                "defects": "|".join(result["defects"]),
                "soft_flags": "|".join(result["soft_flags"]),
                "benchmark_prompt": obj.get("benchmark_prompt", ""),
                "main_goal": obj.get("main_goal", ""),
                "chemical_entity": obj.get("chemical_entity", ""),
                "selected_scenarios": "|".join(
                    split_multi(obj.get("selected_scenarios", []))
                ),
                "distinctive_dimension": obj.get(
                    "distinctive_dimension", ""
                ),
                "model": spec["model"],
                "repaired_at_utc": utcnow(),
            }
            append_csv(path, out, list(out.keys()))
            append_jsonl(
                output_dir/"candidate_lineage.jsonl",
                {
                    "candidate_id": repair_id,
                    "assignment_id": assignment["assignment_id"],
                    "stage": "repair",
                    "parent_candidate_id": cid,
                    "api_meta": meta,
                    "time_utc": utcnow(),
                },
            )
            sync.push(path)
            sync.push(output_dir/"candidate_lineage.jsonl")

            if result["valid"]:
                pass_count += 1
                status = "PASS"
            else:
                fail_count += 1
                status = "FAIL"

        except Exception as exc:
            append_jsonl(
                output_dir/"errors.jsonl",
                {
                    "stage": "repair",
                    "candidate_id": cid,
                    "error": str(exc),
                    "time_utc": utcnow(),
                },
            )
            sync.push(output_dir/"errors.jsonl")
            fail_count += 1
            status = f"ERROR {str(exc)[:100]}"

        completed_now += 1
        _progress(
            "V10 REPAIR",
            completed_before + completed_now,
            total_repairable,
            stage_started,
            f"{item_label} | {status}",
        )

    _stage_done(
        "V10 REPAIR",
        stage_started,
        f"new_pass={pass_count}, new_fail_or_error={fail_count}",
    )

def valid_pool(output_dir):
    output_dir = Path(output_dir)
    rows = []
    cand = pd.read_csv(output_dir/"candidates.csv")
    val = pd.read_csv(output_dir/"validation_results.csv")
    good = val[val["valid"].astype(str).str.lower().isin(["true","1"])]
    merged = cand.merge(good[["candidate_id","assignment_id"]],on=["candidate_id","assignment_id"],how="inner")
    rows.extend(merged.to_dict("records"))

    rp = output_dir/"repairs.csv"
    if rp.exists():
        repair = pd.read_csv(rp)
        repair = repair[repair["valid"].astype(str).str.lower().isin(["true","1"])]
        for _,r in repair.iterrows():
            rows.append({
                "candidate_id":r["repair_candidate_id"],"assignment_id":r["assignment_id"],
                "benchmark_prompt":r["benchmark_prompt"],"main_goal":r["main_goal"],
                "chemical_entity":r["chemical_entity"],"selected_scenarios":r["selected_scenarios"],
                "distinctive_dimension":r["distinctive_dimension"],"generator_model":r["model"],
            })

    fp = output_dir/"refill_candidates.csv"
    if fp.exists():
        refill = pd.read_csv(fp)
        refill = refill[refill["valid"].astype(str).str.lower().isin(["true","1"])]
        for _,r in refill.iterrows():
            rows.append({
                "candidate_id":r["refill_candidate_id"],"assignment_id":r["assignment_id"],
                "benchmark_prompt":r["benchmark_prompt"],"main_goal":r["main_goal"],
                "chemical_entity":r["chemical_entity"],"selected_scenarios":r["selected_scenarios"],
                "distinctive_dimension":r["distinctive_dimension"],"generator_model":r["generator_model"],
            })
    return pd.DataFrame(rows)


def judge_stage(config, prompts, output_dir, sync):
    output_dir = Path(output_dir)
    pool = valid_pool(output_dir)
    if pool.empty:
        raise RuntimeError("No valid candidates available for judging.")

    plan = pd.read_csv(output_dir/"assignments_v10.csv").set_index(
        "assignment_id", drop=False
    )
    selected_path = output_dir/"selected_tasks.csv"
    selected_done = (
        set(pd.read_csv(selected_path)["assignment_id"].astype(str))
        if selected_path.exists()
        else set()
    )
    judgments_path = output_dir/"judgments.csv"

    client = VertexClient(config["project_id"])
    judges = [
        config["models"][name]
        for name in config["judge_roles"]
        if config["models"][name].get("enabled", True)
    ]
    adjudicator = config["models"]["adjudicator"]

    groups = []
    for aid, group in pool.groupby("assignment_id", sort=False):
        aid = str(aid)
        if aid in selected_done or aid not in plan.index:
            continue
        group = group.drop_duplicates(
            subset=["benchmark_prompt"]
        ).reset_index(drop=True)
        if len(group) < 2:
            continue
        groups.append((aid, group))

    total_assignments = len(groups)
    stage_started = _stage_start(
        "V10 JUDGE",
        total_assignments,
        f"judges={len(judges)}, already_selected={len(selected_done)}",
    )

    if total_assignments == 0:
        _stage_done("V10 JUDGE", stage_started, "nothing pending")
        return

    selected_count = 0
    rejected_count = 0

    for assignment_index, (aid, group) in enumerate(groups, start=1):
        print(
            f"[V10 JUDGE] ASSIGNMENT START | {assignment_index}/{total_assignments} | "
            f"{aid} | valid_candidates={len(group)}",
            flush=True,
        )

        best_pair, best_sim = (0, 1), 9.0
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                score = similarity(
                    group.iloc[i]["benchmark_prompt"],
                    group.iloc[j]["benchmark_prompt"],
                )
                if score < best_sim:
                    best_pair, best_sim = (i, j), score

        A = group.iloc[best_pair[0]].to_dict()
        B = group.iloc[best_pair[1]].to_dict()
        assignment = plan.loc[aid]
        records = []

        print(
            f"[V10 JUDGE] pair | {aid} | A={A['candidate_id']} | "
            f"B={B['candidate_id']} | similarity={best_sim:.3f}",
            flush=True,
        )

        for round_no, spec in enumerate(judges, start=1):
            prompt = prompts["pair_judge"].format(
                assignment_json=json.dumps(
                    assignment.to_dict(), ensure_ascii=False
                ),
                candidate_a=json.dumps(
                    candidate_object(A), ensure_ascii=False
                ),
                candidate_b=json.dumps(
                    candidate_object(B), ensure_ascii=False
                ),
            )

            item_label = (
                f"{aid} | judge {round_no}/{len(judges)} | "
                f"A={A['candidate_id']} B={B['candidate_id']}"
            )

            try:
                text_out, meta = _call_with_heartbeat(
                    client,
                    spec,
                    prompts["judge_system"],
                    prompt,
                    stage="V10 JUDGE",
                    item_label=item_label,
                    temperature=0.0,
                    max_tokens=2000,
                )
                result = parse_json_loose(text_out)
                choice = str(
                    result.get("selection", "REJECT_BOTH")
                ).upper()
                selected_id = (
                    A["candidate_id"]
                    if choice == "A"
                    else B["candidate_id"]
                    if choice == "B"
                    else ""
                )

                out = {
                    "judgment_id": f"{aid}-J{round_no}",
                    "assignment_id": aid,
                    "round": round_no,
                    "judge_model": spec["model"],
                    "candidate_a_id": A["candidate_id"],
                    "candidate_b_id": B["candidate_id"],
                    "selection": choice,
                    "selected_candidate_id": selected_id,
                    "reason": result.get("reason", ""),
                    "judged_at_utc": utcnow(),
                }
                append_csv(
                    judgments_path, out, JUDGMENT_COLUMNS
                )
                records.append(
                    {"model": spec["model"], "result": result}
                )
                append_jsonl(
                    output_dir/"judge_lineage.jsonl",
                    {
                        "judgment_id": out["judgment_id"],
                        "api_meta": meta,
                        "time_utc": utcnow(),
                    },
                )
                sync.push(judgments_path)
                sync.push(output_dir/"judge_lineage.jsonl")

                print(
                    f"[V10 JUDGE] judge result | {aid} | "
                    f"{spec['model']} -> {choice}",
                    flush=True,
                )

            except Exception as exc:
                append_jsonl(
                    output_dir/"errors.jsonl",
                    {
                        "stage": "judge",
                        "assignment_id": aid,
                        "model": spec["model"],
                        "error": str(exc),
                        "time_utc": utcnow(),
                    },
                )
                sync.push(output_dir/"errors.jsonl")
                print(
                    f"[V10 JUDGE] judge error | {aid} | "
                    f"{spec['model']} | {str(exc)[:140]}",
                    flush=True,
                )

        choices = [
            str(
                r["result"].get("selection", "REJECT_BOTH")
            ).upper()
            for r in records
        ]

        if (
            len(choices) >= 2
            and len(set(choices)) == 1
            and choices[0] in {"A", "B"}
        ):
            final_choice = choices[0]
            final_reason = "Independent judges agreed."
            print(
                f"[V10 JUDGE] agreement | {aid} | {final_choice}",
                flush=True,
            )

        elif records:
            print(
                f"[V10 JUDGE] disagreement or incomplete consensus | {aid} | "
                f"choices={choices} | adjudication required",
                flush=True,
            )

            prompt = prompts["adjudicator"].format(
                assignment_json=json.dumps(
                    assignment.to_dict(), ensure_ascii=False
                ),
                candidate_a=json.dumps(
                    candidate_object(A), ensure_ascii=False
                ),
                candidate_b=json.dumps(
                    candidate_object(B), ensure_ascii=False
                ),
                judge_records=json.dumps(
                    records, ensure_ascii=False
                ),
            )

            try:
                text_out, meta = _call_with_heartbeat(
                    client,
                    adjudicator,
                    prompts["judge_system"],
                    prompt,
                    stage="V10 ADJUDICATE",
                    item_label=aid,
                    temperature=0.0,
                    max_tokens=adjudicator.get("max_tokens", 1400),
                )
                result = parse_json_loose(text_out)
                final_choice = str(
                    result.get("selection", "REJECT_BOTH")
                ).upper()
                final_reason = str(result.get("reason", ""))

                append_jsonl(
                    output_dir/"judge_lineage.jsonl",
                    {
                        "judgment_id": f"{aid}-ADJ",
                        "api_meta": meta,
                        "time_utc": utcnow(),
                    },
                )
                sync.push(output_dir/"judge_lineage.jsonl")

                print(
                    f"[V10 ADJUDICATE] result | {aid} | {final_choice}",
                    flush=True,
                )

            except Exception as exc:
                final_choice = "REJECT_BOTH"
                final_reason = f"Adjudication error: {exc}"
                print(
                    f"[V10 ADJUDICATE] error | {aid} | {str(exc)[:140]}",
                    flush=True,
                )

        else:
            final_choice = "REJECT_BOTH"
            final_reason = "No completed judge records."

        chosen = (
            A
            if final_choice == "A"
            else B
            if final_choice == "B"
            else None
        )

        if chosen:
            out = {
                "assignment_id": aid,
                "matrix_id": assignment["matrix_id"],
                "hc_id": assignment["hc_id"],
                "hc_category": assignment["hc_category"],
                "hd_id": assignment["hd_id"],
                "hazard_domain": assignment["hazard_domain"],
                "ot_id": assignment["ot_id"],
                "output_type": assignment["output_type"],
                "required_entity": assignment["required_entity"],
                "entity_source": assignment["entity_source"],
                "selected_candidate_id": chosen["candidate_id"],
                "benchmark_prompt": chosen["benchmark_prompt"],
                "main_goal": chosen["main_goal"],
                "chemical_entity": chosen["chemical_entity"],
                "selected_scenarios": chosen["selected_scenarios"],
                "generator_model": chosen["generator_model"],
                "selection_reason": final_reason,
                "selected_at_utc": utcnow(),
            }
            append_csv(selected_path, out, list(out.keys()))
            sync.push(selected_path)
            selected_done.add(aid)
            selected_count += 1
            result_detail = f"SELECTED {chosen['candidate_id']}"

        else:
            append_jsonl(
                output_dir/"rejected_assignments.jsonl",
                {
                    "assignment_id": aid,
                    "reason": final_reason,
                    "time_utc": utcnow(),
                },
            )
            sync.push(output_dir/"rejected_assignments.jsonl")
            rejected_count += 1
            result_detail = "REJECTED"

        _progress(
            "V10 JUDGE",
            assignment_index,
            total_assignments,
            stage_started,
            f"{aid} | {result_detail}",
        )

    _stage_done(
        "V10 JUDGE",
        stage_started,
        f"selected={selected_count}, rejected={rejected_count}",
    )


def refill_stage(config, prompts, output_dir, sync):
    output_dir = Path(output_dir)
    plan = pd.read_csv(output_dir/"assignments_v10.csv")
    selected = (
        pd.read_csv(output_dir/"selected_tasks.csv")
        if (output_dir/"selected_tasks.csv").exists()
        else pd.DataFrame()
    )
    selected_ids = (
        set(selected["assignment_id"].astype(str))
        if not selected.empty
        else set()
    )
    refill_path = output_dir/"refill_candidates.csv"
    done = (
        set(pd.read_csv(refill_path)["assignment_id"].astype(str))
        if refill_path.exists()
        else set()
    )

    pending = [
        assignment
        for _, assignment in plan.iterrows()
        if str(assignment["assignment_id"]) not in selected_ids
        and str(assignment["assignment_id"]) not in done
    ]

    total_refillable = len(pending) + len(done)
    completed_before = len(done)

    client = VertexClient(config["project_id"])
    spec = config["models"]["primary_generator"]
    accepted = current_selected_prompts(output_dir)
    refs = external_refs(output_dir)

    stage_started = _stage_start(
        "V10 REFILL",
        total_refillable,
        f"resume={completed_before} complete, pending={len(pending)}",
    )

    if not pending:
        _stage_done("V10 REFILL", stage_started, "nothing pending")
        return

    completed_now = 0
    pass_count = 0
    fail_count = 0

    for assignment in pending:
        aid = str(assignment["assignment_id"])
        avoid1, avoid3 = diversity_constraints(accepted)

        failure_summary = (
            "No accepted winner after current generation, deterministic "
            "validation, repair, and judging passes."
        )
        prompt = prompts["refill"].format(
            assignment_json=json.dumps(
                assignment.to_dict(), ensure_ascii=False
            ),
            failure_summary=failure_summary,
            avoid_openings=", ".join(avoid1) or "none yet",
            avoid_patterns=" | ".join(avoid3) or "none yet",
        )

        try:
            text_out, meta = _call_with_heartbeat(
                client,
                spec,
                prompts["generator_system"],
                prompt,
                stage="V10 REFILL",
                item_label=aid,
                temperature=0.76,
                max_tokens=spec.get("max_tokens"),
            )
            obj = parse_json_loose(text_out)
            result = deterministic_validate(
                obj, assignment, accepted, refs, config
            )

            out = {
                "refill_candidate_id": aid + "-REFILL1",
                "assignment_id": aid,
                "valid": result["valid"],
                "defects": "|".join(result["defects"]),
                "soft_flags": "|".join(result["soft_flags"]),
                "benchmark_prompt": obj.get("benchmark_prompt", ""),
                "main_goal": obj.get("main_goal", ""),
                "chemical_entity": obj.get("chemical_entity", ""),
                "selected_scenarios": "|".join(
                    split_multi(obj.get("selected_scenarios", []))
                ),
                "distinctive_dimension": obj.get(
                    "distinctive_dimension", ""
                ),
                "generator_model": spec["model"],
                "generated_at_utc": utcnow(),
            }
            append_csv(refill_path, out, list(out.keys()))
            append_jsonl(
                output_dir/"candidate_lineage.jsonl",
                {
                    "candidate_id": aid + "-REFILL1",
                    "assignment_id": aid,
                    "stage": "refill",
                    "api_meta": meta,
                    "time_utc": utcnow(),
                },
            )
            sync.push(refill_path)
            sync.push(output_dir/"candidate_lineage.jsonl")

            if result["valid"]:
                pass_count += 1
                status = "PASS"
            else:
                fail_count += 1
                status = "FAIL"

        except Exception as exc:
            append_jsonl(
                output_dir/"errors.jsonl",
                {
                    "stage": "refill",
                    "assignment_id": aid,
                    "error": str(exc),
                    "time_utc": utcnow(),
                },
            )
            sync.push(output_dir/"errors.jsonl")
            fail_count += 1
            status = f"ERROR {str(exc)[:100]}"

        completed_now += 1
        _progress(
            "V10 REFILL",
            completed_before + completed_now,
            total_refillable,
            stage_started,
            f"{aid} | {status}",
        )

    _stage_done(
        "V10 REFILL",
        stage_started,
        f"new_pass={pass_count}, new_fail_or_error={fail_count}",
    )


def finalize_stage(config, output_dir, sync):
    output_dir = Path(output_dir)
    stage_started = _stage_start("V10 FINALIZE", 4)

    selected_path = output_dir/"selected_tasks.csv"
    if not selected_path.exists():
        raise RuntimeError("No selected tasks yet.")

    selected = pd.read_csv(selected_path)
    plan = pd.read_csv(output_dir/"assignments_v10.csv")

    target = (
        config["test_target"]
        if config["run_type"] == "test"
        else config["pilot_target"]
        if config["run_type"] == "pilot"
        else config["production_target"]
    )

    final = plan[["assignment_id"]].merge(
        selected, on="assignment_id", how="inner"
    ).head(target)

    final_path = output_dir/"final_task_bank.csv"
    final.to_csv(final_path, index=False)
    _progress(
        "V10 FINALIZE", 1, 4, stage_started,
        f"final_task_bank rows={len(final)}"
    )

    coverage = final.groupby(
        ["hc_id", "hd_id", "ot_id"],
        dropna=False,
    ).size().reset_index(name="count")
    coverage.to_csv(
        output_dir/"coverage_report.csv",
        index=False,
    )
    _progress(
        "V10 FINALIZE", 2, 4, stage_started,
        f"coverage rows={len(coverage)}"
    )

    openings = Counter(
        opening(x, 1)
        for x in final[
            "benchmark_prompt"
        ].astype(str)
    )
    pd.DataFrame([
        {
            "opening": key,
            "count": value,
            "share": value / max(1, len(final)),
        }
        for key, value in openings.most_common()
    ]).to_csv(
        output_dir/"diversity_report.csv",
        index=False,
    )
    _progress(
        "V10 FINALIZE", 3, 4, stage_started,
        f"unique_openings={len(openings)}"
    )

    summary = {
        "version": VERSION,
        "namespace": "CBV10C",
        "run_type": config["run_type"],
        "target": target,
        "selected": len(final),
        "completion_label": (
            f"{'COMPLETE' if len(final) >= target else 'CHECKPOINT'}_"
            f"{len(final)}_OF_{target}"
        ),
        "depends_on_prior_chembreak_versions": False,
        "run_signature": config.get(
            "_run_signature", ""
        ),
        "model_roles": {
            role: spec["model"]
            for role, spec in config[
                "models"
            ].items()
            if spec.get("enabled", False)
        },
        "finished_at_utc": utcnow(),
    }

    sp = output_dir/"run_summary.json"
    sp.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    for path in [
        final_path,
        output_dir/"coverage_report.csv",
        output_dir/"diversity_report.csv",
        sp,
    ]:
        sync.push(path)

    _progress(
        "V10 FINALIZE", 4, 4, stage_started,
        summary["completion_label"]
    )
    _stage_done(
        "V10 FINALIZE",
        stage_started,
        summary["completion_label"],
    )
    print(json.dumps(summary, indent=2), flush=True)


def status_stage(output_dir):
    output_dir = Path(output_dir)
    print("[V10 STATUS] current checkpoint files", flush=True)

    for name in [
        "entities_v10.csv",
        "external_reference_behaviors.csv",
        "assignments_v10.csv",
        "candidates.csv",
        "validation_results.csv",
        "repairs.csv",
        "judgments.csv",
        "selected_tasks.csv",
        "refill_candidates.csv",
        "final_task_bank.csv",
    ]:
        path = output_dir/name
        if path.exists():
            try:
                print(
                    f"[V10 STATUS] {name:32s} "
                    f"{len(pd.read_csv(path)):6d} rows",
                    flush=True,
                )
            except Exception:
                print(
                    f"[V10 STATUS] {name:32s} exists",
                    flush=True,
                )
        else:
            print(
                f"[V10 STATUS] {name:32s} not written",
                flush=True,
            )

def run(stage, project_dir, config_path, output_dir):
    project_dir = Path(project_dir)
    output_dir = Path(output_dir)
    config = read_json(config_path)
    prompts,taxonomy = load_assets(project_dir)
    config["_run_signature"] = compute_run_signature(project_dir, config, taxonomy)
    sync = StateSync(output_dir,config.get("gcs_output_uri",""))
    if sync.gcs_uri:
        sync.pull()

    if stage=="preflight":
        preflight(config,output_dir,sync)
    elif stage=="bootstrap":
        bootstrap_sources(project_dir,output_dir,sync)
    elif stage=="plan":
        bootstrap_sources(project_dir,output_dir,sync)
        plan_stage(config,taxonomy,output_dir,sync)
    elif stage=="generate":
        generate_stage(config,prompts,output_dir,sync)
    elif stage=="validate":
        validate_stage(config,output_dir,sync)
    elif stage=="repair":
        repair_stage(config,prompts,output_dir,sync)
    elif stage=="judge":
        judge_stage(config,prompts,output_dir,sync)
    elif stage=="refill":
        refill_stage(config,prompts,output_dir,sync)
    elif stage=="finalize":
        finalize_stage(config,output_dir,sync)
    elif stage=="status":
        status_stage(output_dir)
    elif stage=="all":
        preflight(config,output_dir,sync)
        bootstrap_sources(project_dir,output_dir,sync)
        plan_stage(config,taxonomy,output_dir,sync)
        generate_stage(config,prompts,output_dir,sync)
        validate_stage(config,output_dir,sync)
        repair_stage(config,prompts,output_dir,sync)
        judge_stage(config,prompts,output_dir,sync)
        refill_stage(config,prompts,output_dir,sync)
        judge_stage(config,prompts,output_dir,sync)
        finalize_stage(config,output_dir,sync)
    else:
        raise ValueError(stage)
    sync.push_all()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage",required=True,choices=[
        "preflight","bootstrap","plan","generate","validate","repair",
        "judge","refill","finalize","status","all"
    ])
    p.add_argument("--project-dir",required=True)
    p.add_argument("--config",required=True)
    p.add_argument("--output-dir",required=True)
    args = p.parse_args()
    run(args.stage,args.project_dir,args.config,args.output_dir)

if __name__=="__main__":
    main()
