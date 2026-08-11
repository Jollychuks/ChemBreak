#!/usr/bin/env python3
"""
ChemBreak local/open-weight generator for Google Colab
Package V3

V3 generation architecture:

matrix row
    -> Python loads selected HC/HD/OT definitions
    -> Python assigns a valid scenario
    -> one LLM call generates one candidate
    -> Python overwrites scenario metadata
    -> structural validation
    -> immediate checkpoint to CSV

No paid LLM API is used.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed


OUTPUT_COLUMNS = [
    "candidate_id","matrix_id","candidate_index","hc_id","hc_category",
    "hd_id","hazard_domain","fit","ot_id","output_type","allowed_scenarios",
    "selected_scenarios","benchmark_prompt","main_goal","chemical_entity",
    "distinctive_dimension","generator_model","prompt_version",
    "generation_seed","generated_at_utc",
]

REQUIRED_MATRIX_COLUMNS = {
    "MATRIX_ID","HC_ID","HC_CATEGORY","HD_ID","HAZARD_DOMAIN","FIT",
    "OT_ID","OUTPUT_TYPE","ALLOWED_SCENARIOS","DEFAULT_N_CANDIDATES",
}

REQUIRED_CANDIDATE_KEYS = {
    "benchmark_prompt","main_goal","chemical_entity","distinctive_dimension",
}


def load_config(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_taxonomy(path: str | Path) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("HC", "HD", "OT", "SC"):
        if key not in data:
            raise ValueError(f"taxonomy_definitions.json missing section: {key}")
    return data


def resolve_path(project_dir: str | Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else Path(project_dir) / p


def load_matrix(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_MATRIX_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Matrix missing required columns: {sorted(missing)}")
    return df


def split_scenarios(value: Any) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [x.strip() for x in re.split(r"[|,;]+", text) if x.strip()]


def select_rows(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()

    fit = str(config.get("fit", "ALL")).upper()
    if fit != "ALL":
        out = out[out["FIT"].astype(str).str.upper() == fit]

    matrix_ids = config.get("matrix_ids") or []
    if matrix_ids:
        wanted = {str(x).strip().upper() for x in matrix_ids}
        out = out[out["MATRIX_ID"].astype(str).str.upper().isin(wanted)]

    out = out.reset_index(drop=True)

    start_row = max(int(config.get("start_row", 1)) - 1, 0)
    end_value = config.get("end_row")
    end_row = int(end_value) if end_value not in (None, "", 0) else len(out)

    return out.iloc[start_row:end_row].reset_index(drop=True)


def load_prompt_template(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def stable_row_seed(base_seed: int, matrix_id: str) -> int:
    digest = hashlib.sha256(str(matrix_id).encode("utf-8")).hexdigest()
    return (int(base_seed) + int(digest[:8], 16)) % 2_000_000_000


def bullet_text(items: Sequence[str]) -> str:
    if not items:
        return "NONE"
    return "\n".join(f"- {x}" for x in items)


def scenario_details(
    required_scenarios: Sequence[str],
    taxonomy: Dict[str, Any],
) -> str:
    if not required_scenarios:
        return "NONE"

    scenario_map = taxonomy["SC"]
    return " | ".join(
        f"{sc}: {scenario_map.get(sc, 'Unknown scenario')}"
        for sc in required_scenarios
    )


def build_scenario_plan(
    allowed_scenarios: Sequence[str],
    n: int,
    seed: int,
    *,
    pair_rate: float = 0.0,
) -> List[List[str]]:
    """
    Deterministic Python-controlled scenario assignment.

    Pilot default:
    - candidate 1 uses no scenario
    - later candidates use valid allowed scenarios
    - one scenario per task while pair_rate is 0
    """
    if n < 1:
        return []

    allowed = list(dict.fromkeys(str(x) for x in allowed_scenarios))
    if not allowed:
        return [[] for _ in range(n)]

    rng = random.Random(seed)
    pool = allowed[:]
    rng.shuffle(pool)

    plan: List[List[str]] = [[]]
    cursor = 0

    while len(plan) < n:
        if pair_rate > 0 and len(pool) >= 2 and rng.random() < pair_rate:
            a = pool[cursor % len(pool)]
            b = pool[(cursor + 1) % len(pool)]
            if a != b:
                plan.append([a, b])
                cursor += 2
                continue

        plan.append([pool[cursor % len(pool)]])
        cursor += 1

    return plan[:n]


def format_previous_prompts(prompts: Sequence[str], max_items: int = 8) -> str:
    clean = [str(x).strip() for x in prompts if str(x).strip()]
    if not clean:
        return "NONE"

    lines = []
    for i, text in enumerate(clean[-max_items:], 1):
        compact = re.sub(r"\s+", " ", text)
        if len(compact) > 260:
            compact = compact[:257] + "..."
        lines.append(f"{i}. {compact}")
    return "\n".join(lines)


def render_prompt(
    template: str,
    row: pd.Series,
    taxonomy: Dict[str, Any],
    *,
    required_scenarios: Optional[Sequence[str]] = None,
    previous_prompts: Optional[Sequence[str]] = None,
) -> str:
    required_scenarios = list(required_scenarios or [])
    previous_prompts = list(previous_prompts or [])

    hc_id = str(row["HC_ID"])
    hd_id = str(row["HD_ID"])
    ot_id = str(row["OT_ID"])

    hc = taxonomy["HC"].get(hc_id)
    hd = taxonomy["HD"].get(hd_id)
    ot = taxonomy["OT"].get(ot_id)

    if hc is None:
        raise ValueError(f"No taxonomy definition for {hc_id}")
    if hd is None:
        raise ValueError(f"No taxonomy definition for {hd_id}")
    if ot is None:
        raise ValueError(f"No taxonomy definition for {ot_id}")

    return template.format(
        MATRIX_ID=row["MATRIX_ID"],
        HC_ID=hc_id,
        HC_CATEGORY=row["HC_CATEGORY"],
        HC_DEFINITION=hc["definition"],
        HC_INCLUDE=bullet_text(hc.get("include", [])),
        HC_EXCLUDE=bullet_text(hc.get("exclude", [])),
        HD_ID=hd_id,
        HAZARD_DOMAIN=row["HAZARD_DOMAIN"],
        HD_DEFINITION=hd["definition"],
        OT_ID=ot_id,
        OUTPUT_TYPE=row["OUTPUT_TYPE"],
        OT_DEFINITION=ot["definition"],
        FIT=row["FIT"],
        ALLOWED_SCENARIOS=" | ".join(
            split_scenarios(row["ALLOWED_SCENARIOS"])
        ) or "NONE",
        REQUIRED_SCENARIOS=(
            " | ".join(required_scenarios)
            if required_scenarios
            else "NONE"
        ),
        REQUIRED_SCENARIO_DETAILS=scenario_details(
            required_scenarios, taxonomy
        ),
        PREVIOUS_CANDIDATES=format_previous_prompts(previous_prompts),
    )


def gpu_report() -> Dict[str, Any]:
    report = {
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        report.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "gpu_memory_gb": round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
            ),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        })
    return report


def load_local_model(
    model_id: str,
    *,
    load_in_4bit: bool = True,
    cache_dir: Optional[str] = None,
) -> Tuple[Any, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not detected. In Colab choose Runtime > Change runtime type > GPU."
        )

    print("GPU:", gpu_report())
    print("Loading tokenizer:", model_id)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        use_fast=True,
    )

    kwargs: Dict[str, Any] = {
        "device_map": "auto",
        "cache_dir": cache_dir,
        "low_cpu_mem_usage": True,
    }

    if load_in_4bit:
        compute_dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        kwargs["torch_dtype"] = "auto"

    print("Loading model:", model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Model loaded.")
    return tokenizer, model


def generate_text(
    tokenizer: Any,
    model: Any,
    prompt: str,
    *,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
    seed: int,
    system_message: str = (
        "Follow the ChemBreak benchmark-authoring specification exactly. "
        "Return JSON only. Do not answer the harmful target request."
    ),
) -> str:
    set_seed(seed)

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    do_sample = temperature > 0
    kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    if do_sample:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p

    with torch.inference_mode():
        outputs = model.generate(**kwargs)

    input_len = inputs["input_ids"].shape[-1]
    generated = outputs[0][input_len:]
    return tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.I,
        )
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_response(text: str) -> Dict[str, Any]:
    clean = strip_code_fence(text)

    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        start, end = clean.find("{"), clean.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        data = json.loads(clean[start:end + 1])

    if not isinstance(data, dict):
        raise ValueError("Top-level response must be a JSON object.")

    if "candidates" not in data or not isinstance(
        data["candidates"], list
    ):
        raise ValueError(
            "Response must contain a 'candidates' list."
        )

    return data


def normalize_for_duplicate_check(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def validate_candidate(
    candidate: Dict[str, Any],
    required_scenarios: Sequence[str],
) -> Dict[str, Any]:
    if not isinstance(candidate, dict):
        raise ValueError("Candidate must be a JSON object.")

    missing = REQUIRED_CANDIDATE_KEYS.difference(candidate.keys())
    if missing:
        raise ValueError(
            f"Candidate missing keys: {sorted(missing)}"
        )

    prompt = str(candidate["benchmark_prompt"]).strip()
    goal = str(candidate["main_goal"]).strip()
    entity = str(candidate["chemical_entity"]).strip()
    distinctive = str(candidate["distinctive_dimension"]).strip()

    if not prompt:
        raise ValueError("benchmark_prompt cannot be empty.")
    if not goal:
        raise ValueError("main_goal cannot be empty.")
    if not distinctive:
        raise ValueError(
            "distinctive_dimension cannot be empty."
        )

    return {
        "benchmark_prompt": prompt,
        "main_goal": goal,
        "chemical_entity": entity,
        # Python owns this field.
        "selected_scenarios": list(required_scenarios),
        "distinctive_dimension": distinctive,
    }


def validate_response(
    data: Dict[str, Any],
    *,
    required_scenarios: Sequence[str],
) -> Dict[str, Any]:
    candidates = data["candidates"]

    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly 1 candidate, got {len(candidates)}."
        )

    return validate_candidate(
        candidates[0],
        required_scenarios,
    )


def is_exact_duplicate(
    prompt: str,
    previous_prompts: Sequence[str],
) -> bool:
    current = normalize_for_duplicate_check(prompt)
    return any(
        current == normalize_for_duplicate_check(old)
        for old in previous_prompts
        if str(old).strip()
    )


def generate_with_retries(
    *,
    tokenizer: Any,
    model: Any,
    prompt: str,
    required_scenarios: Sequence[str],
    previous_prompts: Sequence[str],
    config: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    """
    Retry only failures the model actually controls.

    Scenario IDs are controlled by Python and cannot become
    disallowed-scenario validation failures.
    """
    max_retries = int(config.get("max_retries", 5))
    last_error: Optional[Exception] = None
    current_prompt = prompt

    for attempt in range(1, max_retries + 1):
        try:
            text = generate_text(
                tokenizer,
                model,
                current_prompt,
                temperature=float(
                    config.get("temperature", 0.4)
                ),
                top_p=float(
                    config.get("top_p", 0.9)
                ),
                repetition_penalty=float(
                    config.get("repetition_penalty", 1.05)
                ),
                max_new_tokens=int(
                    config.get("max_new_tokens", 1200)
                ),
                seed=seed + attempt - 1,
            )

            data = parse_json_response(text)

            candidate = validate_response(
                data,
                required_scenarios=required_scenarios,
            )

            if is_exact_duplicate(
                candidate["benchmark_prompt"],
                previous_prompts,
            ):
                raise ValueError(
                    "Exact duplicate of a previously accepted "
                    "task from this matrix row."
                )

            return candidate

        except Exception as exc:
            last_error = exc

            print(
                f"      attempt {attempt}/{max_retries} failed: {exc}"
            )

            if attempt >= max_retries:
                break

            current_prompt = prompt + f"""

CORRECTION AFTER INVALID OUTPUT

The previous response failed validation:

{exc}

Regenerate exactly one complete JSON response.

Keep the Python-assigned scenario unchanged.
Return all required fields.
Do not add commentary.
Do not repeat a previously accepted task.
"""

            time.sleep(0.5)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    raise RuntimeError(
        f"Generation failed after {max_retries} attempts: "
        f"{last_error}"
    )


def existing_candidate_ids(output_path: Path) -> set[str]:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return set()

    try:
        df = pd.read_csv(
            output_path,
            usecols=["candidate_id"],
        )
        return set(df["candidate_id"].astype(str))
    except Exception:
        return set()


def ensure_output_version_compatible(
    output_path: Path,
    prompt_version: str,
    *,
    allow_mixed_versions: bool,
) -> None:
    if (
        not output_path.exists()
        or output_path.stat().st_size == 0
        or allow_mixed_versions
    ):
        return

    try:
        df = pd.read_csv(
            output_path,
            usecols=["prompt_version"],
        )
    except Exception:
        return

    versions = {
        str(x)
        for x in df["prompt_version"].dropna().unique()
    }

    if versions and versions != {prompt_version}:
        raise RuntimeError(
            "Existing candidate_tasks.csv contains a different "
            f"prompt version: {sorted(versions)}. "
            f"Current run uses {prompt_version}. "
            "Use a fresh output file or explicitly enable "
            "allow_mixed_prompt_versions."
        )


def load_existing_prompts(
    output_path: Path,
) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}

    if not output_path.exists() or output_path.stat().st_size == 0:
        return result

    try:
        df = pd.read_csv(output_path)
    except Exception:
        return result

    if (
        "matrix_id" not in df.columns
        or "benchmark_prompt" not in df.columns
    ):
        return result

    for matrix_id, group in df.groupby("matrix_id"):
        result[str(matrix_id)] = [
            str(x)
            for x in group["benchmark_prompt"].dropna().tolist()
        ]

    return result


def append_rows(
    output_path: Path,
    rows: List[Dict[str, Any]],
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    exists = (
        output_path.exists()
        and output_path.stat().st_size > 0
    )

    with output_path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=OUTPUT_COLUMNS,
        )
        if not exists:
            writer.writeheader()

        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def append_error(
    error_path: Path,
    payload: Dict[str, Any],
) -> None:
    error_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with error_path.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                payload,
                ensure_ascii=False,
            )
            + "\n"
        )
        f.flush()
        os.fsync(f.fileno())


def update_progress(
    progress_path: Path,
    matrix_id: str,
    target_n: int,
    completed_n: int,
    status: str,
    model_id: str,
) -> None:
    row = {
        "matrix_id": matrix_id,
        "target_candidates": target_n,
        "completed_candidates": completed_n,
        "status": status,
        "model_id": model_id,
        "updated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    if (
        progress_path.exists()
        and progress_path.stat().st_size > 0
    ):
        try:
            df = pd.read_csv(progress_path)
            df = df[
                df["matrix_id"].astype(str)
                != str(matrix_id)
            ]
        except Exception:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    df = pd.concat(
        [df, pd.DataFrame([row])],
        ignore_index=True,
    )
    df.to_csv(progress_path, index=False)


def make_output_row(
    matrix_row: pd.Series,
    candidate: Dict[str, Any],
    absolute_index: int,
    model_id: str,
    prompt_version: str,
    generation_seed: int,
) -> Dict[str, Any]:
    return {
        "candidate_id": (
            f"{matrix_row['MATRIX_ID']}-C"
            f"{absolute_index:04d}"
        ),
        "matrix_id": matrix_row["MATRIX_ID"],
        "candidate_index": absolute_index,
        "hc_id": matrix_row["HC_ID"],
        "hc_category": matrix_row["HC_CATEGORY"],
        "hd_id": matrix_row["HD_ID"],
        "hazard_domain": matrix_row["HAZARD_DOMAIN"],
        "fit": matrix_row["FIT"],
        "ot_id": matrix_row["OT_ID"],
        "output_type": matrix_row["OUTPUT_TYPE"],
        "allowed_scenarios": "|".join(
            split_scenarios(
                matrix_row["ALLOWED_SCENARIOS"]
            )
        ),
        "selected_scenarios": "|".join(
            candidate["selected_scenarios"]
        ),
        "benchmark_prompt": candidate["benchmark_prompt"],
        "main_goal": candidate["main_goal"],
        "chemical_entity": candidate["chemical_entity"],
        "distinctive_dimension": (
            candidate["distinctive_dimension"]
        ),
        "generator_model": model_id,
        "prompt_version": prompt_version,
        "generation_seed": generation_seed,
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }


def run_generation(
    project_dir: str | Path,
    config: Dict[str, Any],
    tokenizer: Any,
    model: Any,
) -> Path:
    """
    Generate one candidate per model call.

    Python chooses the scenario.
    Every accepted candidate is saved immediately.
    """
    project_dir = Path(project_dir)

    matrix_path = resolve_path(
        project_dir,
        config["matrix_file"],
    )
    taxonomy_path = resolve_path(
        project_dir,
        config["taxonomy_file"],
    )
    prompt_path = resolve_path(
        project_dir,
        config["prompt_file"],
    )
    output_path = resolve_path(
        project_dir,
        config["output_file"],
    )
    progress_path = resolve_path(
        project_dir,
        config["progress_file"],
    )
    error_path = resolve_path(
        project_dir,
        config["error_log_file"],
    )

    matrix = select_rows(
        load_matrix(matrix_path),
        config,
    )
    taxonomy = load_taxonomy(taxonomy_path)
    template = load_prompt_template(prompt_path)

    if matrix.empty:
        raise ValueError(
            "No generation-matrix rows matched "
            "the current configuration."
        )

    n_override = config.get("n_per_row")
    base_seed = int(config.get("seed", 42))
    resume = bool(config.get("resume", True))
    model_id = str(config["model_id"])
    prompt_version = str(config["prompt_version"])
    pair_rate = float(
        config.get("scenario_pair_rate", 0.0)
    )

    ensure_output_version_compatible(
        output_path,
        prompt_version,
        allow_mixed_versions=bool(
            config.get(
                "allow_mixed_prompt_versions",
                False,
            )
        ),
    )

    seen_ids = (
        existing_candidate_ids(output_path)
        if resume
        else set()
    )

    previous_prompts = (
        load_existing_prompts(output_path)
        if resume
        else {}
    )

    print(f"Selected matrix rows: {len(matrix)}")
    print(f"Candidates per row: {n_override}")
    print("Generation mode: ONE candidate per model call")
    print("Scenario control: PYTHON")
    print(f"Scenario pair rate: {pair_rate}")
    print(f"Prompt version: {prompt_version}")
    print(f"Output: {output_path}")
    print(f"Resume: {resume}")

    for _, row in tqdm(
        matrix.iterrows(),
        total=len(matrix),
        desc="Matrix rows",
    ):
        total_for_row = (
            int(n_override)
            if n_override not in (None, "", 0)
            else int(row["DEFAULT_N_CANDIDATES"])
        )

        matrix_id = str(row["MATRIX_ID"])
        allowed = split_scenarios(
            row["ALLOWED_SCENARIOS"]
        )
        row_seed = stable_row_seed(
            base_seed,
            matrix_id,
        )

        plan = build_scenario_plan(
            allowed,
            total_for_row,
            row_seed,
            pair_rate=pair_rate,
        )

        row_previous = previous_prompts.setdefault(
            matrix_id,
            [],
        )

        completed = sum(
            1
            for idx in range(1, total_for_row + 1)
            if (
                f"{matrix_id}-C{idx:04d}"
                in seen_ids
            )
        )

        print(
            f"\n{matrix_id} | "
            f"{row['HC_ID']} + "
            f"{row['HD_ID']} + "
            f"{row['OT_ID']} | "
            f"completed {completed}/{total_for_row}"
        )

        for absolute_index in range(
            1,
            total_for_row + 1,
        ):
            candidate_id = (
                f"{matrix_id}-C"
                f"{absolute_index:04d}"
            )

            if candidate_id in seen_ids:
                continue

            required_scenarios = plan[
                absolute_index - 1
            ]

            candidate_seed = (
                row_seed
                + absolute_index * 100
            )

            assigned_label = (
                required_scenarios
                if required_scenarios
                else "NONE"
            )

            print(
                f"  {candidate_id} | "
                f"assigned scenario: {assigned_label}"
            )

            prompt = render_prompt(
                template,
                row,
                taxonomy,
                required_scenarios=required_scenarios,
                previous_prompts=row_previous,
            )

            try:
                candidate = generate_with_retries(
                    tokenizer=tokenizer,
                    model=model,
                    prompt=prompt,
                    required_scenarios=required_scenarios,
                    previous_prompts=row_previous,
                    config=config,
                    seed=candidate_seed,
                )

                output_row = make_output_row(
                    row,
                    candidate,
                    absolute_index,
                    model_id,
                    prompt_version,
                    candidate_seed,
                )

                append_rows(
                    output_path,
                    [output_row],
                )

                seen_ids.add(candidate_id)
                row_previous.append(
                    candidate["benchmark_prompt"]
                )
                completed += 1

                update_progress(
                    progress_path,
                    matrix_id,
                    total_for_row,
                    completed,
                    (
                        "COMPLETE"
                        if completed == total_for_row
                        else "IN_PROGRESS"
                    ),
                    model_id,
                )

                print(
                    f"    saved {candidate_id}"
                )

            except Exception as exc:
                append_error(
                    error_path,
                    {
                        "candidate_id": candidate_id,
                        "matrix_id": matrix_id,
                        "hc_id": row["HC_ID"],
                        "hd_id": row["HD_ID"],
                        "ot_id": row["OT_ID"],
                        "required_scenarios": (
                            required_scenarios
                        ),
                        "error": str(exc),
                        "time_utc": datetime.now(
                            timezone.utc
                        ).isoformat(),
                    },
                )

                update_progress(
                    progress_path,
                    matrix_id,
                    total_for_row,
                    completed,
                    "PARTIAL_ERROR",
                    model_id,
                )

                print(
                    f"    ERROR logged for "
                    f"{candidate_id}: {exc}"
                )

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\nGeneration pass finished.")
    print("Candidates:", output_path)
    print("Progress:", progress_path)
    print("Errors:", error_path)

    return output_path


def generate_test_candidates(
    row: pd.Series,
    *,
    n: int,
    template: str,
    taxonomy: Dict[str, Any],
    tokenizer: Any,
    model: Any,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Test the same V3 scenario-control path used by
    the full generation job.
    """
    allowed = split_scenarios(
        row["ALLOWED_SCENARIOS"]
    )

    row_seed = stable_row_seed(
        int(config.get("seed", 42)),
        str(row["MATRIX_ID"]),
    )

    plan = build_scenario_plan(
        allowed,
        n,
        row_seed,
        pair_rate=float(
            config.get("scenario_pair_rate", 0.0)
        ),
    )

    results: List[Dict[str, Any]] = []
    previous: List[str] = []

    for idx in range(1, n + 1):
        assigned = plan[idx - 1]

        print(
            f"Test candidate {idx}/{n} | "
            f"Python-assigned scenario: "
            f"{assigned if assigned else 'NONE'}"
        )

        prompt = render_prompt(
            template,
            row,
            taxonomy,
            required_scenarios=assigned,
            previous_prompts=previous,
        )

        candidate = generate_with_retries(
            tokenizer=tokenizer,
            model=model,
            prompt=prompt,
            required_scenarios=assigned,
            previous_prompts=previous,
            config=config,
            seed=row_seed + idx * 100,
        )

        previous.append(
            candidate["benchmark_prompt"]
        )
        results.append(candidate)

    return results


def print_candidate_summary(
    output_path: str | Path,
) -> None:
    path = Path(output_path)

    if not path.exists() or path.stat().st_size == 0:
        print("No candidate output exists yet.")
        return

    df = pd.read_csv(path)

    print("Total candidates:", len(df))
    print(
        "Unique matrix rows:",
        df["matrix_id"].nunique(),
    )

    print("\nBy HC:")
    print(
        df["hc_id"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nBy HD:")
    print(
        df["hd_id"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nBy OT:")
    print(
        df["ot_id"]
        .value_counts()
        .sort_index()
        .to_string()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-dir",
        required=True,
    )
    parser.add_argument(
        "--config",
        default="run_config.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    config = load_config(
        resolve_path(
            project_dir,
            args.config,
        )
    )

    matrix = select_rows(
        load_matrix(
            resolve_path(
                project_dir,
                config["matrix_file"],
            )
        ),
        config,
    )

    taxonomy = load_taxonomy(
        resolve_path(
            project_dir,
            config["taxonomy_file"],
        )
    )

    template = load_prompt_template(
        resolve_path(
            project_dir,
            config["prompt_file"],
        )
    )

    if args.dry_run:
        if matrix.empty:
            raise ValueError("No rows selected.")

        row = matrix.iloc[0]

        plan = build_scenario_plan(
            split_scenarios(
                row["ALLOWED_SCENARIOS"]
            ),
            1,
            stable_row_seed(
                int(config.get("seed", 42)),
                str(row["MATRIX_ID"]),
            ),
        )

        print(
            render_prompt(
                template,
                row,
                taxonomy,
                required_scenarios=plan[0],
                previous_prompts=[],
            )
        )
        return

    tokenizer, model = load_local_model(
        config["model_id"],
        load_in_4bit=bool(
            config.get("load_in_4bit", True)
        ),
        cache_dir=(
            config.get("hf_cache_dir")
            or None
        ),
    )

    run_generation(
        project_dir,
        config,
        tokenizer,
        model,
    )


if __name__ == "__main__":
    main()
