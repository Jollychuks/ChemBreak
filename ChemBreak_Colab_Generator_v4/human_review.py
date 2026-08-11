from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from chembreak_common import (
    bullet_text,
    parse_pipe_list,
    resolve_path,
    scenario_details,
)
from judge_multimodel import (
    SCORE_COLUMNS,
    decision_from_scores,
)


def create_blinded_human_sample(
    project_dir: str | Path,
    config: Dict[str, Any],
    taxonomy: Dict[str, Any],
):
    project_dir = Path(
        project_dir
    )

    out_dir = resolve_path(
        project_dir,
        config["output_dir"],
    )

    candidate_path = resolve_path(
        project_dir,
        config["candidate_file"],
    )

    if not candidate_path.exists():
        raise FileNotFoundError(
            "Run generation before "
            "creating a human-review sample."
        )

    candidates = pd.read_csv(
        candidate_path
    ).fillna("")

    candidates = candidates[
        candidates[
            "experiment_id"
        ].astype(str)
        == str(
            config["experiment_id"]
        )
    ].copy()

    if candidates.empty:
        raise ValueError(
            "No candidates match "
            "the current experiment."
        )

    seed = int(
        config[
            "human_review"
        ]["seed"]
    )

    rng = random.Random(
        seed
    )

    chosen_rows: List[
        pd.Series
    ] = []

    if config[
        "human_review"
    ].get(
        "sample_one_per_hc_per_generator",
        True,
    ):
        for (
            _family,
            _hc,
        ), group in candidates.groupby(
            [
                "generator_family_id",
                "hc_id",
            ]
        ):
            indices = list(
                group.index
            )

            chosen_index = (
                rng.choice(
                    indices
                )
            )

            chosen_rows.append(
                candidates.loc[
                    chosen_index
                ]
            )
    else:
        sample_n = min(
            36,
            len(candidates),
        )

        chosen_rows = [
            row
            for _, row
            in candidates.sample(
                n=sample_n,
                random_state=seed,
            ).iterrows()
        ]

    # Shuffle the selected rows again so the
    # reviewer does not see generator blocks.
    rng.shuffle(
        chosen_rows
    )

    review_rows = []
    key_rows = []

    for sequence, row in enumerate(
        chosen_rows,
        1,
    ):
        review_id = (
            f"HR-{sequence:04d}"
        )

        hc = taxonomy["HC"][
            str(row["hc_id"])
        ]
        hd = taxonomy["HD"][
            str(row["hd_id"])
        ]
        ot = taxonomy["OT"][
            str(row["ot_id"])
        ]

        selected = parse_pipe_list(
            row[
                "selected_scenarios"
            ]
        )

        review_row = {
            "review_id":
                review_id,
            "matrix_id":
                row["matrix_id"],
            "hc_id":
                row["hc_id"],
            "hc_category":
                row[
                    "hc_category"
                ],
            "hc_definition":
                hc["definition"],
            "hc_inclusions":
                " | ".join(
                    hc.get(
                        "include",
                        [],
                    )
                ),
            "hc_exclusions":
                " | ".join(
                    hc.get(
                        "exclude",
                        [],
                    )
                ),
            "hd_id":
                row["hd_id"],
            "hazard_domain":
                row[
                    "hazard_domain"
                ],
            "hd_definition":
                hd["definition"],
            "ot_id":
                row["ot_id"],
            "output_type":
                row[
                    "output_type"
                ],
            "ot_definition":
                ot["definition"],
            "selected_scenarios":
                row[
                    "selected_scenarios"
                ],
            "scenario_details":
                scenario_details(
                    taxonomy,
                    selected,
                ),
            "benchmark_prompt":
                row[
                    "benchmark_prompt"
                ],
        }

        for col in SCORE_COLUMNS:
            review_row[
                f"human_{col}"
            ] = ""

        review_row[
            "human_decision"
        ] = ""

        review_row[
            "human_notes"
        ] = ""

        review_rows.append(
            review_row
        )

        key_rows.append({
            "review_id":
                review_id,
            "candidate_id":
                row[
                    "candidate_id"
                ],
            "generator_family_id":
                row[
                    "generator_family_id"
                ],
            "generator_family_name":
                row[
                    "generator_family_name"
                ],
            "generator_model_id":
                row[
                    "generator_model_id"
                ],
        })

    blinded_path = (
        out_dir
        / "human_review_blinded.csv"
    )

    key_path = (
        out_dir
        / "human_review_key.csv"
    )

    pd.DataFrame(
        review_rows
    ).to_csv(
        blinded_path,
        index=False,
    )

    pd.DataFrame(
        key_rows
    ).to_csv(
        key_path,
        index=False,
    )

    return (
        blinded_path,
        key_path,
    )


def evaluate_judges_against_human(
    project_dir: str | Path,
    config: Dict[str, Any],
    completed_review_file:
        str | Path | None = None,
):
    """
    Compare all four model judges to blinded human scores.

    If all eight human scores are present, Python derives the
    reference ACCEPT/REVISE/REJECT label with the exact same
    decision rule used for model judges. A manually entered
    human_decision is retained only for consistency checking.
    """
    project_dir = Path(project_dir)

    out_dir = resolve_path(
        project_dir,
        config["output_dir"],
    )

    review_path = (
        Path(completed_review_file)
        if completed_review_file
        else (
            out_dir
            / "human_review_blinded.csv"
        )
    )

    key_path = (
        out_dir
        / "human_review_key.csv"
    )

    judgment_path = resolve_path(
        project_dir,
        config["judgment_file"],
    )

    review = pd.read_csv(
        review_path
    ).fillna("")

    key = pd.read_csv(
        key_path
    ).fillna("")

    judgments = pd.read_csv(
        judgment_path
    ).fillna("")

    # Keep rows with all eight human score fields completed.
    completed_mask = pd.Series(
        True,
        index=review.index,
    )

    for col in SCORE_COLUMNS:
        completed_mask &= pd.to_numeric(
            review[f"human_{col}"],
            errors="coerce",
        ).notna()

    review = review[
        completed_mask
    ].copy()

    if review.empty:
        raise ValueError(
            "No human-review rows have all "
            "eight human score fields completed."
        )

    reference_decisions = []
    human_overall_scores = []
    manual_consistency = []

    for _, row in review.iterrows():
        scores = {
            col: int(
                pd.to_numeric(
                    row[f"human_{col}"],
                    errors="raise",
                )
            )
            for col in SCORE_COLUMNS
        }

        reference = decision_from_scores(
            scores,
            config,
        )

        reference_decisions.append(
            reference
        )

        human_overall_scores.append(
            sum(scores.values())
            / len(SCORE_COLUMNS)
        )

        manual = str(
            row.get(
                "human_decision",
                "",
            )
        ).strip().upper()

        if manual:
            manual_consistency.append(
                manual == reference
            )
        else:
            manual_consistency.append(
                None
            )

    review[
        "reference_decision"
    ] = reference_decisions

    review[
        "human_overall_quality_score"
    ] = human_overall_scores

    review[
        "manual_decision_matches_scores"
    ] = manual_consistency

    mapped = review.merge(
        key,
        on="review_id",
        how="left",
    )

    compared = mapped.merge(
        judgments,
        on="candidate_id",
        how="left",
        suffixes=(
            "_human",
            "_judge",
        ),
    )

    judge_rows = []

    for (
        judge_family_id,
        group,
    ) in compared.groupby(
        "judge_family_id"
    ):
        decision_agreement = (
            group[
                "reference_decision"
            ].astype(str)
            == group[
                "validator_decision"
            ].astype(str)
        ).mean()

        score_errors = []

        for col in SCORE_COLUMNS:
            human_col = f"human_{col}"

            human_values = pd.to_numeric(
                group[human_col],
                errors="coerce",
            )

            judge_values = pd.to_numeric(
                group[col],
                errors="coerce",
            )

            valid = (
                human_values.notna()
                & judge_values.notna()
            )

            if valid.any():
                score_errors.extend(
                    (
                        human_values[valid]
                        - judge_values[valid]
                    )
                    .abs()
                    .tolist()
                )

        judge_rows.append({
            "judge_family_id":
                judge_family_id,
            "judge_family_name":
                group[
                    "judge_family_name"
                ].iloc[0],
            "judge_model_id":
                group[
                    "judge_model_id"
                ].iloc[0],
            "human_labeled_candidates":
                group[
                    "candidate_id"
                ].nunique(),
            "human_decision_exact_agreement":
                decision_agreement,
            "mean_absolute_score_error":
                (
                    sum(score_errors)
                    / len(score_errors)
                    if score_errors
                    else float("nan")
                ),
        })

    judge_vs_human = pd.DataFrame(
        judge_rows
    ).sort_values(
        by=[
            "human_decision_exact_agreement",
            "mean_absolute_score_error",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    if not judge_vs_human.empty:
        judge_vs_human.insert(
            0,
            "human_calibrated_judge_rank",
            range(
                1,
                len(judge_vs_human) + 1,
            ),
        )

    judge_path = (
        out_dir
        / "judge_vs_human.csv"
    )

    judge_vs_human.to_csv(
        judge_path,
        index=False,
    )

    # Direct human comparison of the four generator families.
    generator_rows = []

    for (
        generator_family_id,
        group,
    ) in mapped.groupby(
        "generator_family_id"
    ):
        decisions = group[
            "reference_decision"
        ].astype(str)

        generator_rows.append({
            "generator_family_id":
                generator_family_id,
            "generator_family_name":
                group[
                    "generator_family_name"
                ].iloc[0],
            "generator_model_id":
                group[
                    "generator_model_id"
                ].iloc[0],
            "human_labeled_candidates":
                len(group),
            "human_mean_quality_score":
                pd.to_numeric(
                    group[
                        "human_overall_quality_score"
                    ],
                    errors="coerce",
                ).mean(),
            "human_accept_rate":
                decisions.eq("ACCEPT").mean(),
            "human_revise_rate":
                decisions.eq("REVISE").mean(),
            "human_reject_rate":
                decisions.eq("REJECT").mean(),
        })

    generator_vs_human = (
        pd.DataFrame(
            generator_rows
        )
        .sort_values(
            by=[
                "human_mean_quality_score",
                "human_accept_rate",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    if not generator_vs_human.empty:
        generator_vs_human.insert(
            0,
            "human_calibrated_generator_rank",
            range(
                1,
                len(generator_vs_human) + 1,
            ),
        )

    generator_path = (
        out_dir
        / "generator_vs_human.csv"
    )

    generator_vs_human.to_csv(
        generator_path,
        index=False,
    )

    # Save the scored blinded review with Python-derived reference labels.
    reference_path = (
        out_dir
        / "human_review_reference_scored.csv"
    )

    review.to_csv(
        reference_path,
        index=False,
    )

    return {
        "judge_vs_human":
            judge_path,
        "generator_vs_human":
            generator_path,
        "human_reference_scored":
            reference_path,
    }

