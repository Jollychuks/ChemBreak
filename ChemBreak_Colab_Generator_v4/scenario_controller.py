from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from chembreak_common import (
    load_json,
    resolve_path,
    select_rows,
    sha256_file,
    split_scenarios,
    stable_int_seed,
    utc_now,
    write_json,
)


PLAN_COLUMNS = [
    "matrix_id",
    "candidate_index",
    "scenario_assignment",
    "scenario_assignment_count",
    "scenario_seed",
    "scenario_plan_version",
]


def build_row_plan(
    allowed_scenarios: List[str],
    n: int,
    seed: int,
    include_baseline: bool,
    pair_rate: float = 0.0,
) -> List[List[str]]:
    """
    Create a reproducible balanced plan.

    With pair_rate=0:
    - optional candidate 1 is the NONE baseline
    - each allowed scenario is used once before
      any scenario repeats
    - order is deterministic per matrix row
    """
    if n < 1:
        return []

    allowed = list(
        dict.fromkeys(
            str(x)
            for x in allowed_scenarios
        )
    )

    rng = random.Random(seed)
    pool = allowed[:]
    rng.shuffle(pool)

    plan: List[List[str]] = []

    if include_baseline:
        plan.append([])

    if not allowed:
        while len(plan) < n:
            plan.append([])
        return plan[:n]

    cursor = 0

    while len(plan) < n:
        if (
            pair_rate > 0
            and len(pool) >= 2
            and rng.random() < pair_rate
        ):
            a = pool[cursor % len(pool)]
            b = pool[(cursor + 1) % len(pool)]

            if a != b:
                plan.append([a, b])
                cursor += 2
                continue

        plan.append([
            pool[cursor % len(pool)]
        ])
        cursor += 1

    return plan[:n]


def plan_signature(
    config: Dict[str, Any],
    matrix_path: Path,
) -> Dict[str, Any]:
    return {
        "scenario_plan_version":
            config["scenario_plan_version"],
        "scenario_seed":
            config["scenario_seed"],
        "n_per_row":
            config["n_per_row"],
        "include_unconditioned_baseline":
            config["include_unconditioned_baseline"],
        "scenario_pair_rate":
            config["scenario_pair_rate"],
        "matrix_ids":
            config.get("matrix_ids") or [],
        "fit":
            config.get("fit", "ALL"),
        "start_row":
            config.get("start_row", 1),
        "end_row":
            config.get("end_row"),
        "matrix_sha256":
            sha256_file(matrix_path),
    }


def create_scenario_plan(
    project_dir: str | Path,
    config: Dict[str, Any],
    matrix: pd.DataFrame,
) -> pd.DataFrame:
    selected = select_rows(matrix, config)
    rows = []

    for _, row in selected.iterrows():
        matrix_id = str(row["MATRIX_ID"])

        allowed = split_scenarios(
            row["ALLOWED_SCENARIOS"]
        )

        row_seed = stable_int_seed(
            config["scenario_seed"],
            matrix_id,
            config["scenario_plan_version"],
        )

        row_plan = build_row_plan(
            allowed,
            int(config["n_per_row"]),
            row_seed,
            bool(
                config[
                    "include_unconditioned_baseline"
                ]
            ),
            float(
                config.get(
                    "scenario_pair_rate",
                    0.0,
                )
            ),
        )

        for idx, assigned in enumerate(
            row_plan,
            1,
        ):
            rows.append({
                "matrix_id": matrix_id,
                "candidate_index": idx,
                "scenario_assignment":
                    "|".join(assigned),
                "scenario_assignment_count":
                    len(assigned),
                "scenario_seed": row_seed,
                "scenario_plan_version":
                    config[
                        "scenario_plan_version"
                    ],
            })

    return pd.DataFrame(
        rows,
        columns=PLAN_COLUMNS,
    )


def ensure_scenario_plan(
    project_dir: str | Path,
    config: Dict[str, Any],
    matrix: pd.DataFrame,
) -> pd.DataFrame:
    project_dir = Path(project_dir)

    matrix_path = resolve_path(
        project_dir,
        config["matrix_file"],
    )

    plan_path = resolve_path(
        project_dir,
        config["scenario_plan_file"],
    )

    manifest_path = resolve_path(
        project_dir,
        config["scenario_plan_manifest_file"],
    )

    signature = plan_signature(
        config,
        matrix_path,
    )

    if (
        plan_path.exists()
        and manifest_path.exists()
    ):
        existing_manifest = load_json(
            manifest_path
        )

        if (
            existing_manifest.get("signature")
            != signature
        ):
            raise RuntimeError(
                "Existing scenario plan was created "
                "with different matrix or scenario "
                "settings. Use a fresh output "
                "directory or restore the matching "
                "configuration."
            )

        return pd.read_csv(
            plan_path
        ).fillna("")

    plan = create_scenario_plan(
        project_dir,
        config,
        matrix,
    )

    plan_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plan.to_csv(
        plan_path,
        index=False,
    )

    write_json(
        manifest_path,
        {
            "created_at_utc": utc_now(),
            "signature": signature,
            "rows": len(plan),
        },
    )

    return plan
