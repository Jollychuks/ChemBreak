
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

VERSION = "8.0-cloud"
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
    def __init__(self, project_id):
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        self.project_id = project_id
        self.session = AuthorizedSession(creds)

    @staticmethod
    def _host(location):
        return "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"

    def call(self, spec, system_text, user_text, temperature=None, max_tokens=None):
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
                "systemInstruction": {"parts":[{"text":system_text}]},
                "contents": [{"role":"user","parts":[{"text":user_text}]}],
                "generationConfig": {
                    "temperature": t,
                    "topP": 0.95,
                    "maxOutputTokens": m,
                    "responseMimeType": "application/json",
                }
            }
        elif style == "openai_compatible":
            version = spec.get("api_version", "v1")
            url = (
                f"https://{host}/{version}/projects/{self.project_id}/locations/{location}/"
                "endpoints/openapi/chat/completions"
            )
            payload = {
                "model": model,
                "messages": [
                    {"role":"system","content":system_text},
                    {"role":"user","content":user_text},
                ],
                "temperature": t,
                "max_tokens": m,
                "stream": False,
            }
            if spec.get("reasoning_effort"):
                payload["reasoning_effort"] = spec["reasoning_effort"]
        else:
            raise ValueError(f"Unsupported api_style: {style}")

        response = self.session.post(url, json=payload, timeout=300)
        if response.status_code >= 400:
            raise RuntimeError(f"{model} HTTP {response.status_code}: {response.text[:1600]}")
        data = response.json()

        if style == "gemini":
            candidates = data.get("candidates") or []
            if not candidates:
                raise RuntimeError(f"{model} returned no candidates")
            parts = candidates[0].get("content",{}).get("parts",[])
            text = "".join(str(p.get("text","")) for p in parts if "text" in p)
            usage = data.get("usageMetadata", {})
        else:
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"{model} returned no choices")
            text = choices[0].get("message",{}).get("content","")
            usage = data.get("usage",{})

        if not str(text).strip():
            raise RuntimeError(f"{model} returned empty text")
        return str(text).strip(), {
            "model": model,
            "location": location,
            "api_style": style,
            "elapsed_seconds": round(time.time()-started,3),
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
    taxonomy = read_json(project_dir/"taxonomy"/"taxonomy_v8.json")
    return prompts, taxonomy

def bootstrap_sources(project_dir, output_dir, sync):
    output_dir = Path(output_dir)
    entities_path = output_dir/"entities_v8.csv"
    refs_path = output_dir/"external_reference_behaviors.csv"
    if entities_path.exists() and refs_path.exists():
        print("Bootstrap already complete.")
        return

    # Fresh entity source from upstream ChemSafety.
    substances_url = "https://raw.githubusercontent.com/IDEA-XL/ChemSafety/main/substances.json"
    r = requests.get(substances_url, timeout=60)
    r.raise_for_status()
    substances = r.json()

    entity_rows = []
    for group, items in substances.items():
        for item in items:
            name = str(item.get("Name","")).strip()
            if not name:
                continue
            if group == "Drugs":
                allowed_domains = "HD1"
            elif group == "Explosives":
                allowed_domains = "HD4"
            else:
                # The upstream group combines chemical-warfare and poisonous materials,
                # so V8 does not pretend the source supplies a finer label.
                allowed_domains = "HD2|HD3"
            entity_rows.append({
                "entity_id": f"CS-{len(entity_rows)+1:03d}",
                "entity_name": name,
                "source_group": group,
                "allowed_hazard_domains": allowed_domains,
                "entity_type": "chemical substance",
                "source": "IDEA-XL/ChemSafety substances.json",
            })

    # High-level system/class targets for coverage not represented by that small source list.
    # These are benchmark taxonomy targets, not copied prior ChemBreak tasks.
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
                "entity_id": f"V8-{hd}-{len([x for x in entity_rows if x['allowed_hazard_domains']==hd])+1:02d}",
                "entity_name": name,
                "source_group": "V8 taxonomy system target",
                "allowed_hazard_domains": hd,
                "entity_type": typ,
                "source": "ChemBreak V8 Cloud taxonomy specification",
            })

    pd.DataFrame(entity_rows).to_csv(entities_path, index=False)

    # Fresh upstream HarmBench reference, used only for external similarity filtering.
    hb_url = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
    r = requests.get(hb_url, timeout=60)
    r.raise_for_status()
    hb = pd.read_csv(io.StringIO(r.text))
    if "SemanticCategory" in hb.columns:
        hb = hb[hb["SemanticCategory"].astype(str).eq("chemical_biological")].copy()
    keep = [c for c in ["Behavior","BehaviorID","SemanticCategory","FunctionalCategory"] if c in hb.columns]
    hb[keep].to_csv(refs_path, index=False)

    sync.push(entities_path)
    sync.push(refs_path)
    print(f"Created {len(entity_rows)} fresh entity/source rows.")
    print(f"Created {len(hb)} external similarity references.")

def plan_stage(config, taxonomy, output_dir, sync):
    output_dir = Path(output_dir)
    plan_path = output_dir/"assignments_v8.csv"
    if plan_path.exists():
        print("Plan already exists:", plan_path)
        return

    entities = pd.read_csv(output_dir/"entities_v8.csv")
    run_type = config["run_type"]
    if run_type == "test":
        final_target, reserve = config["test_target"], 0
    elif run_type == "pilot":
        final_target, reserve = config["pilot_target"], config["pilot_reserve"]
    elif run_type == "production":
        final_target, reserve = config["production_target"], config["production_reserve"]
    else:
        raise ValueError("run_type must be test, pilot, or production")
    total = final_target + reserve

    if run_type == "test":
        hc_targets = {hc:1 for hc in taxonomy["capabilities"]}
        hd_targets = scaled_targets(taxonomy["pilot_hd_targets"], total)
    else:
        hc_targets = scaled_targets(taxonomy["pilot_hc_targets"], total)
        hd_targets = scaled_targets(taxonomy["pilot_hd_targets"], total)

    rng = random.Random(config["seed"])
    hc_remaining = hc_targets.copy()
    hd_remaining = hd_targets.copy()
    ot_counts = Counter()
    entity_cursor = defaultdict(int)
    rows = []

    def entity_pool_for_hd(hd):
        mask = entities["allowed_hazard_domains"].astype(str).apply(
            lambda x: hd in split_multi(x.replace("|", ","))
        )
        return entities[mask].reset_index(drop=True)

    for i in range(total):
        hcs = [k for k,v in hc_remaining.items() if v > 0]
        if not hcs:
            hcs = list(taxonomy["capabilities"])
        hc = max(hcs, key=lambda k: (hc_remaining.get(k,0)/max(1,hc_targets.get(k,1)), rng.random()))

        compatible = taxonomy["hc_hd"][hc]
        hds = [hd for hd in compatible if hd_remaining.get(hd,0) > 0 and compatible[hd] in {"strong","selective"}]
        if not hds:
            hds = [hd for hd,fit in compatible.items() if fit in {"strong","selective"}]
        hd = max(
            hds,
            key=lambda x: (
                3 if compatible[x] == "strong" else 1,
                hd_remaining.get(x,0)/max(1,hd_targets.get(x,1)),
                rng.random(),
            )
        )

        allowed_ots = list(taxonomy["hc_ot"][hc])
        if rng.random() < 0.08:
            allowed_ots = list(dict.fromkeys(allowed_ots + ["OT14"]))
        ot = min(allowed_ots, key=lambda x: (ot_counts[(hc,x)], rng.random()))
        ot_counts[(hc,ot)] += 1

        pool = entity_pool_for_hd(hd)
        if pool.empty:
            raise RuntimeError(f"No entity/system source rows available for {hd}")
        erow = pool.iloc[entity_cursor[hd] % len(pool)]
        entity_cursor[hd] += 1

        if ot == "OT15":
            preferred = ["SC12","SC15","SC13"]
        elif ot == "OT14":
            preferred = ["SC11","SC15","SC14"]
        elif hc == "HC5":
            preferred = ["SC13","SC06","SC05","SC14"]
        elif hc == "HC8":
            preferred = ["SC08","SC11","SC15"]
        elif hc == "HC7":
            preferred = ["SC09","SC14","SC07"]
        elif hc == "HC9":
            preferred = ["SC10","SC14","SC11"]
        else:
            preferred = ["SC01","SC02","SC03","SC14"]

        scenarios = []
        if rng.random() < 0.80:
            scenarios.append(rng.choice(preferred))
        if rng.random() < 0.22:
            remaining = [x for x in preferred if x not in scenarios]
            if remaining:
                scenarios.append(rng.choice(remaining))

        aid = f"CBV8C-{i+1:04d}"
        rows.append({
            "assignment_id": aid,
            "matrix_id": f"V8C-{hc}-{hd}-{ot}",
            "hc_id": hc,
            "hc_category": taxonomy["capabilities"][hc]["name"],
            "hc_definition": taxonomy["capabilities"][hc]["definition"],
            "hd_id": hd,
            "hazard_domain": taxonomy["hazard_domains"][hd],
            "fit": compatible[hd],
            "ot_id": ot,
            "output_type": taxonomy["output_types"][ot],
            "required_entity": erow["entity_name"],
            "entity_source": erow["source"],
            "allowed_scenarios": "|".join(scenarios),
            "assigned_scenario": "|".join(scenarios),
            "request_form": REQUEST_FORMS[i % len(REQUEST_FORMS)],
            "diversity_focus": DIVERSITY_FOCI[(i // len(REQUEST_FORMS)) % len(DIVERSITY_FOCI)],
            "is_reserve": i >= final_target,
        })
        hc_remaining[hc] = max(0, hc_remaining.get(hc,0)-1)
        hd_remaining[hd] = max(0, hd_remaining.get(hd,0)-1)

    plan = pd.DataFrame(rows)
    plan.to_csv(plan_path, index=False)
    manifest = {
        "version": VERSION,
        "run_type": run_type,
        "namespace": "CBV8C",
        "fresh_v8_cloud_plan": True,
        "depends_on_prior_chembreak_versions": False,
        "final_target": final_target,
        "reserve": reserve,
        "planned_assignments": len(plan),
        "taxonomy_sha256": hashlib.sha256(
            json.dumps(taxonomy, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "created_at_utc": utcnow(),
    }
    mp = output_dir/"plan_manifest.json"
    mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    sync.push(plan_path)
    sync.push(mp)
    print(json.dumps(manifest, indent=2))

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
    client = VertexClient(config["project_id"])
    rows = []
    for role,spec in config["models"].items():
        if not spec.get("enabled",True):
            rows.append({"role":role,"model":spec["model"],"status":"DISABLED","detail":""})
            continue
        if spec["api_style"] == "model_garden_view_code":
            rows.append({"role":role,"model":spec["model"],"status":"DISABLED","detail":"Paste current View Code model ID before enabling."})
            continue
        try:
            text,meta = client.call(spec, "Return JSON only.", 'Return exactly {"ok": true}.', temperature=0.0, max_tokens=100)
            parse_json_loose(text)
            rows.append({"role":role,"model":spec["model"],"status":"OK","detail":""})
        except Exception as exc:
            rows.append({"role":role,"model":spec["model"],"status":"ERROR","detail":str(exc)[:1200]})
    path = Path(output_dir)/"preflight_models.csv"
    pd.DataFrame(rows).to_csv(path,index=False)
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
    plan = pd.read_csv(output_dir/"assignments_v8.csv")
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

    for row_i, assignment in plan.iterrows():
        prompt = render_generator(prompts, assignment, accepted)
        for slot,role in enumerate(roles, start=1):
            cid = f"{assignment['assignment_id']}-C{slot:02d}"
            if cid in existing:
                continue
            spec = config["models"][role]
            try:
                text,meta = client.call(spec,prompts["generator_system"],prompt)
                obj = parse_json_loose(text)
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
                    "selected_scenarios": "|".join(split_multi(obj.get("selected_scenarios",[]))),
                    "benchmark_prompt": obj.get("benchmark_prompt",""),
                    "main_goal": obj.get("main_goal",""),
                    "chemical_entity": obj.get("chemical_entity",""),
                    "distinctive_dimension": obj.get("distinctive_dimension",""),
                    "generator_model": spec["model"],
                    "prompt_version": "CB-V8-CLOUD-GEN-1",
                    "generation_seed": config["seed"] + row_i*10 + slot,
                    "generated_at_utc": utcnow(),
                }
                append_csv(candidates_path,row,CANDIDATE_COLUMNS)
                append_jsonl(output_dir/"candidate_lineage.jsonl",{
                    "candidate_id":cid,"assignment_id":assignment["assignment_id"],
                    "stage":"generate","role":role,"api_meta":meta,
                    "response_sha256":hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "time_utc":utcnow(),
                })
                sync.push(candidates_path)
                sync.push(output_dir/"candidate_lineage.jsonl")
                existing.add(cid)
                print(assignment["assignment_id"],role,"saved")
            except Exception as exc:
                append_jsonl(output_dir/"errors.jsonl",{
                    "stage":"generate","assignment_id":assignment["assignment_id"],
                    "role":role,"model":spec["model"],"error":str(exc),"time_utc":utcnow()
                })
                sync.push(output_dir/"errors.jsonl")
                print(assignment["assignment_id"],role,"ERROR",str(exc)[:220])

def validate_stage(config, output_dir, sync):
    output_dir = Path(output_dir)
    plan = pd.read_csv(output_dir/"assignments_v8.csv").set_index("assignment_id",drop=False)
    candidates = pd.read_csv(output_dir/"candidates.csv")
    path = output_dir/"validation_results.csv"
    done = set(pd.read_csv(path)["candidate_id"].astype(str)) if path.exists() else set()
    accepted = current_selected_prompts(output_dir)
    refs = external_refs(output_dir)

    for _,row in candidates.iterrows():
        cid = str(row["candidate_id"])
        if cid in done:
            continue
        assignment = plan.loc[str(row["assignment_id"])]
        obj = candidate_object(row)
        result = deterministic_validate(obj,assignment,accepted,refs,config)
        out = {
            "candidate_id":cid,"assignment_id":assignment["assignment_id"],
            "valid":result["valid"],"defects":"|".join(result["defects"]),
            "soft_flags":"|".join(result["soft_flags"]),
            "word_count":result["word_count"],
            "duplicate_score":result["duplicate_score"],
            "external_reference_score":result["external_reference_score"],
            "validated_at_utc":utcnow(),
        }
        append_csv(path,out,VALIDATION_COLUMNS)
        sync.push(path)
        print(cid,"PASS" if result["valid"] else "FAIL",out["defects"][:180])

def repair_stage(config, prompts, output_dir, sync):
    output_dir = Path(output_dir)
    val = pd.read_csv(output_dir/"validation_results.csv")
    cand = pd.read_csv(output_dir/"candidates.csv").set_index("candidate_id",drop=False)
    plan = pd.read_csv(output_dir/"assignments_v8.csv").set_index("assignment_id",drop=False)
    path = output_dir/"repairs.csv"
    done = set(pd.read_csv(path)["original_candidate_id"].astype(str)) if path.exists() else set()
    client = VertexClient(config["project_id"])
    spec = config["models"]["repair_model"]
    accepted = current_selected_prompts(output_dir)
    refs = external_refs(output_dir)

    for _,vr in val.iterrows():
        if str(vr["valid"]).lower() in {"true","1"}:
            continue
        cid = str(vr["candidate_id"])
        if cid in done or cid not in cand.index:
            continue
        row = cand.loc[cid]
        assignment = plan.loc[str(vr["assignment_id"])]
        avoid1,avoid3 = diversity_constraints(accepted)
        prompt = prompts["repair"].format(
            assignment_json=json.dumps(assignment.to_dict(),ensure_ascii=False),
            candidate_json=json.dumps(candidate_object(row),ensure_ascii=False),
            defects=vr.get("defects",""),
            soft_flags=vr.get("soft_flags",""),
            avoid_openings=", ".join(avoid1) or "none yet",
            avoid_patterns=" | ".join(avoid3) or "none yet",
        )
        try:
            text,meta = client.call(spec,prompts["repair_system"],prompt,temperature=0.30)
            obj = parse_json_loose(text)
            result = deterministic_validate(obj,assignment,accepted,refs,config)
            repair_id = cid+"-R1"
            out = {
                "original_candidate_id":cid,"repair_candidate_id":repair_id,
                "assignment_id":assignment["assignment_id"],
                "valid":result["valid"],"defects":"|".join(result["defects"]),
                "soft_flags":"|".join(result["soft_flags"]),
                "benchmark_prompt":obj.get("benchmark_prompt",""),
                "main_goal":obj.get("main_goal",""),
                "chemical_entity":obj.get("chemical_entity",""),
                "selected_scenarios":"|".join(split_multi(obj.get("selected_scenarios",[]))),
                "distinctive_dimension":obj.get("distinctive_dimension",""),
                "model":spec["model"],"repaired_at_utc":utcnow(),
            }
            append_csv(path,out,list(out.keys()))
            append_jsonl(output_dir/"candidate_lineage.jsonl",{
                "candidate_id":repair_id,"assignment_id":assignment["assignment_id"],
                "stage":"repair","parent_candidate_id":cid,"api_meta":meta,"time_utc":utcnow()
            })
            sync.push(path)
            sync.push(output_dir/"candidate_lineage.jsonl")
            print(cid,"repair","PASS" if result["valid"] else "FAIL")
        except Exception as exc:
            append_jsonl(output_dir/"errors.jsonl",{
                "stage":"repair","candidate_id":cid,"error":str(exc),"time_utc":utcnow()
            })
            sync.push(output_dir/"errors.jsonl")

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
    plan = pd.read_csv(output_dir/"assignments_v8.csv").set_index("assignment_id",drop=False)
    selected_path = output_dir/"selected_tasks.csv"
    selected_done = set(pd.read_csv(selected_path)["assignment_id"].astype(str)) if selected_path.exists() else set()
    judgments_path = output_dir/"judgments.csv"
    client = VertexClient(config["project_id"])
    judges = [config["models"][name] for name in config["judge_roles"] if config["models"][name].get("enabled",True)]
    adjudicator = config["models"]["adjudicator"]

    for aid,group in pool.groupby("assignment_id",sort=False):
        aid = str(aid)
        if aid in selected_done or aid not in plan.index:
            continue
        group = group.drop_duplicates(subset=["benchmark_prompt"]).reset_index(drop=True)
        if len(group) < 2:
            continue

        # Judge the two most textually different valid survivors.
        best_pair, best_sim = (0,1), 9.0
        for i in range(len(group)):
            for j in range(i+1,len(group)):
                s = similarity(group.iloc[i]["benchmark_prompt"],group.iloc[j]["benchmark_prompt"])
                if s < best_sim:
                    best_pair,best_sim = (i,j),s
        A = group.iloc[best_pair[0]].to_dict()
        B = group.iloc[best_pair[1]].to_dict()
        assignment = plan.loc[aid]
        records = []

        for round_no,spec in enumerate(judges,start=1):
            prompt = prompts["pair_judge"].format(
                assignment_json=json.dumps(assignment.to_dict(),ensure_ascii=False),
                candidate_a=json.dumps(candidate_object(A),ensure_ascii=False),
                candidate_b=json.dumps(candidate_object(B),ensure_ascii=False),
            )
            try:
                text,meta = client.call(spec,prompts["judge_system"],prompt,temperature=0.0,max_tokens=2000)
                result = parse_json_loose(text)
                choice = str(result.get("selection","REJECT_BOTH")).upper()
                selected_id = A["candidate_id"] if choice=="A" else B["candidate_id"] if choice=="B" else ""
                out = {
                    "judgment_id":f"{aid}-J{round_no}","assignment_id":aid,"round":round_no,
                    "judge_model":spec["model"],"candidate_a_id":A["candidate_id"],
                    "candidate_b_id":B["candidate_id"],"selection":choice,
                    "selected_candidate_id":selected_id,"reason":result.get("reason",""),
                    "judged_at_utc":utcnow(),
                }
                append_csv(judgments_path,out,JUDGMENT_COLUMNS)
                records.append({"model":spec["model"],"result":result})
                append_jsonl(output_dir/"judge_lineage.jsonl",{
                    "judgment_id":out["judgment_id"],"api_meta":meta,"time_utc":utcnow()
                })
                sync.push(judgments_path)
                sync.push(output_dir/"judge_lineage.jsonl")
            except Exception as exc:
                append_jsonl(output_dir/"errors.jsonl",{
                    "stage":"judge","assignment_id":aid,"model":spec["model"],
                    "error":str(exc),"time_utc":utcnow()
                })
                sync.push(output_dir/"errors.jsonl")

        choices = [str(r["result"].get("selection","REJECT_BOTH")).upper() for r in records]
        if len(choices)>=2 and len(set(choices))==1 and choices[0] in {"A","B"}:
            final_choice = choices[0]
            final_reason = "Independent judges agreed."
        elif records:
            prompt = prompts["adjudicator"].format(
                assignment_json=json.dumps(assignment.to_dict(),ensure_ascii=False),
                candidate_a=json.dumps(candidate_object(A),ensure_ascii=False),
                candidate_b=json.dumps(candidate_object(B),ensure_ascii=False),
                judge_records=json.dumps(records,ensure_ascii=False),
            )
            try:
                text,meta = client.call(adjudicator,prompts["judge_system"],prompt,temperature=0.0,max_tokens=1400)
                result = parse_json_loose(text)
                final_choice = str(result.get("selection","REJECT_BOTH")).upper()
                final_reason = str(result.get("reason",""))
                append_jsonl(output_dir/"judge_lineage.jsonl",{
                    "judgment_id":f"{aid}-ADJ","api_meta":meta,"time_utc":utcnow()
                })
                sync.push(output_dir/"judge_lineage.jsonl")
            except Exception as exc:
                final_choice = "REJECT_BOTH"
                final_reason = f"Adjudication error: {exc}"
        else:
            final_choice = "REJECT_BOTH"
            final_reason = "No completed judge records."

        chosen = A if final_choice=="A" else B if final_choice=="B" else None
        if chosen:
            out = {
                "assignment_id":aid,"matrix_id":assignment["matrix_id"],
                "hc_id":assignment["hc_id"],"hc_category":assignment["hc_category"],
                "hd_id":assignment["hd_id"],"hazard_domain":assignment["hazard_domain"],
                "ot_id":assignment["ot_id"],"output_type":assignment["output_type"],
                "required_entity":assignment["required_entity"],"entity_source":assignment["entity_source"],
                "selected_candidate_id":chosen["candidate_id"],
                "benchmark_prompt":chosen["benchmark_prompt"],"main_goal":chosen["main_goal"],
                "chemical_entity":chosen["chemical_entity"],
                "selected_scenarios":chosen["selected_scenarios"],
                "generator_model":chosen["generator_model"],
                "selection_reason":final_reason,"selected_at_utc":utcnow(),
            }
            append_csv(selected_path,out,list(out.keys()))
            sync.push(selected_path)
            selected_done.add(aid)
            print(aid,"SELECTED",chosen["candidate_id"])
        else:
            append_jsonl(output_dir/"rejected_assignments.jsonl",{
                "assignment_id":aid,"reason":final_reason,"time_utc":utcnow()
            })
            sync.push(output_dir/"rejected_assignments.jsonl")
            print(aid,"REJECTED")

def refill_stage(config, prompts, output_dir, sync):
    output_dir = Path(output_dir)
    plan = pd.read_csv(output_dir/"assignments_v8.csv")
    selected = pd.read_csv(output_dir/"selected_tasks.csv") if (output_dir/"selected_tasks.csv").exists() else pd.DataFrame()
    selected_ids = set(selected["assignment_id"].astype(str)) if not selected.empty else set()
    refill_path = output_dir/"refill_candidates.csv"
    done = set(pd.read_csv(refill_path)["assignment_id"].astype(str)) if refill_path.exists() else set()
    client = VertexClient(config["project_id"])
    spec = config["models"]["primary_generator"]
    accepted = current_selected_prompts(output_dir)
    refs = external_refs(output_dir)

    for _,assignment in plan.iterrows():
        aid = str(assignment["assignment_id"])
        if aid in selected_ids or aid in done:
            continue
        avoid1,avoid3 = diversity_constraints(accepted)
        failure_summary = "No accepted winner after current generation, deterministic validation, repair, and judging passes."
        prompt = prompts["refill"].format(
            assignment_json=json.dumps(assignment.to_dict(),ensure_ascii=False),
            failure_summary=failure_summary,
            avoid_openings=", ".join(avoid1) or "none yet",
            avoid_patterns=" | ".join(avoid3) or "none yet",
        )
        try:
            text,meta = client.call(spec,prompts["generator_system"],prompt,temperature=0.76)
            obj = parse_json_loose(text)
            result = deterministic_validate(obj,assignment,accepted,refs,config)
            out = {
                "refill_candidate_id":aid+"-REFILL1","assignment_id":aid,
                "valid":result["valid"],"defects":"|".join(result["defects"]),
                "soft_flags":"|".join(result["soft_flags"]),
                "benchmark_prompt":obj.get("benchmark_prompt",""),
                "main_goal":obj.get("main_goal",""),
                "chemical_entity":obj.get("chemical_entity",""),
                "selected_scenarios":"|".join(split_multi(obj.get("selected_scenarios",[]))),
                "distinctive_dimension":obj.get("distinctive_dimension",""),
                "generator_model":spec["model"],"generated_at_utc":utcnow(),
            }
            append_csv(refill_path,out,list(out.keys()))
            append_jsonl(output_dir/"candidate_lineage.jsonl",{
                "candidate_id":aid+"-REFILL1","assignment_id":aid,
                "stage":"refill","api_meta":meta,"time_utc":utcnow()
            })
            sync.push(refill_path)
            sync.push(output_dir/"candidate_lineage.jsonl")
            print(aid,"refill","PASS" if result["valid"] else "FAIL")
        except Exception as exc:
            append_jsonl(output_dir/"errors.jsonl",{
                "stage":"refill","assignment_id":aid,"error":str(exc),"time_utc":utcnow()
            })
            sync.push(output_dir/"errors.jsonl")

def finalize_stage(config, output_dir, sync):
    output_dir = Path(output_dir)
    selected_path = output_dir/"selected_tasks.csv"
    if not selected_path.exists():
        raise RuntimeError("No selected tasks yet.")
    selected = pd.read_csv(selected_path)
    plan = pd.read_csv(output_dir/"assignments_v8.csv")
    target = (
        config["test_target"] if config["run_type"]=="test"
        else config["pilot_target"] if config["run_type"]=="pilot"
        else config["production_target"]
    )
    final = plan[["assignment_id"]].merge(selected,on="assignment_id",how="inner").head(target)
    final_path = output_dir/"final_task_bank.csv"
    final.to_csv(final_path,index=False)

    coverage = final.groupby(["hc_id","hd_id","ot_id"],dropna=False).size().reset_index(name="count")
    coverage.to_csv(output_dir/"coverage_report.csv",index=False)

    openings = Counter(opening(x,1) for x in final["benchmark_prompt"].astype(str))
    pd.DataFrame([
        {"opening":k,"count":v,"share":v/max(1,len(final))}
        for k,v in openings.most_common()
    ]).to_csv(output_dir/"diversity_report.csv",index=False)

    summary = {
        "version":VERSION,"namespace":"CBV8C","run_type":config["run_type"],
        "target":target,"selected":len(final),
        "completion_label":f"{'COMPLETE' if len(final)>=target else 'CHECKPOINT'}_{len(final)}_OF_{target}",
        "depends_on_prior_chembreak_versions":False,
        "model_roles":{
            role:spec["model"] for role,spec in config["models"].items() if spec.get("enabled",False)
        },
        "finished_at_utc":utcnow(),
    }
    sp = output_dir/"run_summary.json"
    sp.write_text(json.dumps(summary,indent=2),encoding="utf-8")
    for p in [final_path,output_dir/"coverage_report.csv",output_dir/"diversity_report.csv",sp]:
        sync.push(p)
    print(json.dumps(summary,indent=2))

def status_stage(output_dir):
    output_dir = Path(output_dir)
    for name in [
        "entities_v8.csv","external_reference_behaviors.csv","assignments_v8.csv",
        "candidates.csv","validation_results.csv","repairs.csv","judgments.csv",
        "selected_tasks.csv","refill_candidates.csv","final_task_bank.csv"
    ]:
        p = output_dir/name
        if p.exists():
            try:
                print(f"{name:32s} {len(pd.read_csv(p)):6d} rows")
            except Exception:
                print(name,"exists")
        else:
            print(f"{name:32s} not written")

def run(stage, project_dir, config_path, output_dir):
    project_dir = Path(project_dir)
    output_dir = Path(output_dir)
    config = read_json(config_path)
    prompts,taxonomy = load_assets(project_dir)
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
