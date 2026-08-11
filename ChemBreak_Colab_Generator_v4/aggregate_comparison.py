from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd

from chembreak_common import (
    resolve_path,
    select_rows,
)
from judge_multimodel import (
    SCORE_COLUMNS,
)


DECISION_ORDER = [
    "ACCEPT",
    "REVISE",
    "REJECT",
]


def _consensus_decision(
    decisions: Iterable[str],
    minimum_matching: int,
) -> str:
    series = pd.Series(
        list(decisions)
    ).dropna().astype(str)

    if series.empty:
        return "NO_CONSENSUS"

    counts = (
        series.value_counts()
    )

    top_decision = str(
        counts.index[0]
    )
    top_count = int(
        counts.iloc[0]
    )

    if top_count >= minimum_matching:
        return top_decision

    return "NO_CONSENSUS"


def _cohen_kappa(
    a: pd.Series,
    b: pd.Series,
) -> float:
    pair = pd.DataFrame({
        "a": a.astype(str),
        "b": b.astype(str),
    }).dropna()

    if pair.empty:
        return float("nan")

    observed = (
        pair["a"]
        .eq(pair["b"])
        .mean()
    )

    labels = sorted(
        set(pair["a"])
        | set(pair["b"])
    )

    expected = 0.0

    for label in labels:
        pa = (
            pair["a"]
            .eq(label)
            .mean()
        )
        pb = (
            pair["b"]
            .eq(label)
            .mean()
        )
        expected += pa * pb

    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else float("nan")

    return (
        observed - expected
    ) / (
        1.0 - expected
    )


def aggregate_results(
    project_dir: str | Path,
    config: Dict[str, Any],
    matrix: pd.DataFrame | None = None,
) -> Dict[str, Path]:
    project_dir = Path(
        project_dir
    )

    out_dir = resolve_path(
        project_dir,
        config["output_dir"],
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_path = resolve_path(
        project_dir,
        config["candidate_file"],
    )
    judgment_path = resolve_path(
        project_dir,
        config["judgment_file"],
    )

    if not candidate_path.exists():
        raise FileNotFoundError(
            "Candidate file not found."
        )

    if not judgment_path.exists():
        raise FileNotFoundError(
            "Judgment file not found."
        )

    candidates = pd.read_csv(
        candidate_path
    ).fillna("")

    judgments = pd.read_csv(
        judgment_path
    ).fillna("")

    candidates = candidates[
        candidates[
            "experiment_id"
        ].astype(str)
        == str(
            config["experiment_id"]
        )
    ].copy()

    judgments = judgments[
        judgments[
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

    if judgments.empty:
        raise ValueError(
            "No judgments match "
            "the current experiment."
        )

    minimum_matching = int(
        config[
            "consensus_rule"
        ][
            "minimum_matching_judges"
        ]
    )

    # ---------------------------------------------------------------
    # Candidate-level consensus
    # ---------------------------------------------------------------
    candidate_rows = []

    for candidate_id, group in (
        judgments.groupby(
            "candidate_id"
        )
    ):
        matches = candidates[
            candidates[
                "candidate_id"
            ].astype(str)
            == str(candidate_id)
        ]

        if matches.empty:
            continue

        base = matches.iloc[0]

        decisions = group[
            "validator_decision"
        ].astype(str)

        row = {
            "candidate_id":
                candidate_id,
            "generator_family_id":
                base[
                    "generator_family_id"
                ],
            "generator_family_name":
                base[
                    "generator_family_name"
                ],
            "matrix_id":
                base["matrix_id"],
            "hc_id":
                base["hc_id"],
            "hd_id":
                base["hd_id"],
            "ot_id":
                base["ot_id"],
            "judge_count":
                len(group),
            "consensus_decision":
                _consensus_decision(
                    decisions,
                    minimum_matching,
                ),
            "decision_agreement_fraction":
                (
                    decisions
                    .value_counts()
                    .iloc[0]
                    / len(group)
                ),
        }

        for col in SCORE_COLUMNS:
            row[
                f"mean_{col}"
            ] = pd.to_numeric(
                group[col],
                errors="coerce",
            ).mean()

        row[
            "mean_overall_quality_score"
        ] = pd.to_numeric(
            group[
                "overall_quality_score"
            ],
            errors="coerce",
        ).mean()

        candidate_rows.append(
            row
        )

    candidate_consensus = (
        pd.DataFrame(
            candidate_rows
        )
    )

    candidate_consensus_path = (
        out_dir
        / "candidate_consensus.csv"
    )

    candidate_consensus.to_csv(
        candidate_consensus_path,
        index=False,
    )

    # ---------------------------------------------------------------
    # Expected generation count per family
    # ---------------------------------------------------------------
    if matrix is not None:
        selected_rows = len(
            select_rows(
                matrix,
                config,
            )
        )
    else:
        selected_rows = len(
            set(
                candidates[
                    "matrix_id"
                ].astype(str)
            )
        )

    expected_per_generator = (
        selected_rows
        * int(
            config["n_per_row"]
        )
    )

    judge_family_count = len(
        config["judge_families"]
    )

    expected_cross_judgments_per_generator = (
        expected_per_generator
        * max(
            judge_family_count - 1,
            0,
        )
    )

    # ---------------------------------------------------------------
    # Generator summary
    # Primary automated comparison excludes self-family judge.
    # ---------------------------------------------------------------
    generator_rows = []

    for (
        generator_family_id,
        candidate_group,
    ) in candidates.groupby(
        "generator_family_id"
    ):
        candidate_ids = set(
            candidate_group[
                "candidate_id"
            ].astype(str)
        )

        judged = judgments[
            judgments[
                "candidate_id"
            ].astype(str)
            .isin(candidate_ids)
        ].copy()

        cross = judged[
            judged[
                "judge_family_id"
            ].astype(str)
            != str(
                generator_family_id
            )
        ].copy()

        self_judged = judged[
            judged[
                "judge_family_id"
            ].astype(str)
            == str(
                generator_family_id
            )
        ].copy()

        consensus = (
            candidate_consensus[
                candidate_consensus[
                    "generator_family_id"
                ].astype(str)
                == str(
                    generator_family_id
                )
            ]
        )

        row = {
            "generator_family_id":
                generator_family_id,
            "generator_family_name":
                candidate_group[
                    "generator_family_name"
                ].iloc[0],
            "generator_model_id":
                candidate_group[
                    "generator_model_id"
                ].iloc[0],
            "expected_candidate_count":
                expected_per_generator,
            "candidate_count":
                len(
                    candidate_group
                ),
            "generation_completion_rate":
                (
                    len(
                        candidate_group
                    )
                    / expected_per_generator
                    if expected_per_generator
                    else float("nan")
                ),
            "first_attempt_generation_rate":
                (
                    pd.to_numeric(
                        candidate_group[
                            "generation_attempts"
                        ],
                        errors="coerce",
                    )
                    .eq(1)
                    .mean()
                ),
            "all_judges_mean_quality":
                pd.to_numeric(
                    judged[
                        "overall_quality_score"
                    ],
                    errors="coerce",
                ).mean(),
            "cross_family_judges_mean_quality":
                pd.to_numeric(
                    cross[
                        "overall_quality_score"
                    ],
                    errors="coerce",
                ).mean(),
            "self_judge_mean_quality":
                pd.to_numeric(
                    self_judged[
                        "overall_quality_score"
                    ],
                    errors="coerce",
                ).mean(),
            "cross_family_judgment_count":
                len(cross),
            "expected_cross_family_judgments":
                expected_cross_judgments_per_generator,
            "cross_family_judgment_coverage":
                (
                    len(cross)
                    / expected_cross_judgments_per_generator
                    if expected_cross_judgments_per_generator
                    else float("nan")
                ),
            "all_judges_accept_rate":
                judged[
                    "validator_decision"
                ].astype(str)
                .eq("ACCEPT")
                .mean(),
            "cross_family_judges_accept_rate":
                cross[
                    "validator_decision"
                ].astype(str)
                .eq("ACCEPT")
                .mean(),
            "consensus_accept_rate":
                consensus[
                    "consensus_decision"
                ].astype(str)
                .eq("ACCEPT")
                .mean(),
            "consensus_revise_rate":
                consensus[
                    "consensus_decision"
                ].astype(str)
                .eq("REVISE")
                .mean(),
            "consensus_reject_rate":
                consensus[
                    "consensus_decision"
                ].astype(str)
                .eq("REJECT")
                .mean(),
            "no_consensus_rate":
                consensus[
                    "consensus_decision"
                ].astype(str)
                .eq("NO_CONSENSUS")
                .mean(),
        }

        for col in SCORE_COLUMNS:
            row[
                f"cross_mean_{col}"
            ] = pd.to_numeric(
                cross[col],
                errors="coerce",
            ).mean()

        generator_rows.append(
            row
        )

    generator_summary = (
        pd.DataFrame(
            generator_rows
        )
    )

    if not generator_summary.empty:
        # All-judge rank uses the identical four-judge panel
        # for every generator.
        all_ranked = (
            generator_summary
            .sort_values(
                by=[
                    "all_judges_mean_quality",
                    "all_judges_accept_rate",
                    "generation_completion_rate",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
            )
            .reset_index(drop=True)
        )

        all_rank_map = {
            str(row["generator_family_id"]): rank
            for rank, (_, row)
            in enumerate(
                all_ranked.iterrows(),
                1,
            )
        }

        # Cross-family rank removes each generator's
        # own-family judge. The panel therefore differs
        # by generator, so this is reported separately.
        cross_ranked = (
            generator_summary
            .sort_values(
                by=[
                    "cross_family_judges_mean_quality",
                    "cross_family_judges_accept_rate",
                    "generation_completion_rate",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
            )
            .reset_index(drop=True)
        )

        cross_rank_map = {
            str(row["generator_family_id"]): rank
            for rank, (_, row)
            in enumerate(
                cross_ranked.iterrows(),
                1,
            )
        }

        generator_summary[
            "automated_rank_all_judges"
        ] = (
            generator_summary[
                "generator_family_id"
            ]
            .astype(str)
            .map(all_rank_map)
        )

        generator_summary[
            "automated_rank_cross_family"
        ] = (
            generator_summary[
                "generator_family_id"
            ]
            .astype(str)
            .map(cross_rank_map)
        )

        generator_summary = (
            generator_summary
            .sort_values(
                by=[
                    "automated_rank_all_judges",
                    "automated_rank_cross_family",
                ]
            )
            .reset_index(drop=True)
        )

        leading = [
            "automated_rank_all_judges",
            "automated_rank_cross_family",
        ]

        remaining = [
            c
            for c in generator_summary.columns
            if c not in leading
        ]

        generator_summary = (
            generator_summary[
                leading + remaining
            ]
        )

    generator_summary_path = (
        out_dir
        / "generator_summary.csv"
    )

    generator_summary.to_csv(
        generator_summary_path,
        index=False,
    )

    # ---------------------------------------------------------------
    # Judge behavior summary
    # Descriptive only until compared to human labels.
    # ---------------------------------------------------------------
    judge_rows = []

    for (
        judge_family_id,
        group,
    ) in judgments.groupby(
        "judge_family_id"
    ):
        same = group[
            group[
                "generator_family_id"
            ].astype(str)
            == str(
                judge_family_id
            )
        ]

        cross = group[
            group[
                "generator_family_id"
            ].astype(str)
            != str(
                judge_family_id
            )
        ]

        same_mean = pd.to_numeric(
            same[
                "overall_quality_score"
            ],
            errors="coerce",
        ).mean()

        cross_mean = pd.to_numeric(
            cross[
                "overall_quality_score"
            ],
            errors="coerce",
        ).mean()

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
            "judgment_count":
                len(group),
            "mean_quality_score_assigned":
                pd.to_numeric(
                    group[
                        "overall_quality_score"
                    ],
                    errors="coerce",
                ).mean(),
            "accept_rate":
                group[
                    "validator_decision"
                ].astype(str)
                .eq("ACCEPT")
                .mean(),
            "revise_rate":
                group[
                    "validator_decision"
                ].astype(str)
                .eq("REVISE")
                .mean(),
            "reject_rate":
                group[
                    "validator_decision"
                ].astype(str)
                .eq("REJECT")
                .mean(),
            "same_family_mean_quality_assigned":
                same_mean,
            "other_family_mean_quality_assigned":
                cross_mean,
            "same_minus_other_family_score":
                (
                    same_mean
                    - cross_mean
                ),
        })

    judge_summary = pd.DataFrame(
        judge_rows
    )

    judge_summary_path = (
        out_dir
        / "judge_summary.csv"
    )

    judge_summary.to_csv(
        judge_summary_path,
        index=False,
    )

    # ---------------------------------------------------------------
    # Generator x judge matrix
    # ---------------------------------------------------------------
    cross_rows = []

    for (
        gen,
        judge,
    ), group in judgments.groupby(
        [
            "generator_family_id",
            "judge_family_id",
        ]
    ):
        row = {
            "generator_family_id":
                gen,
            "judge_family_id":
                judge,
            "judgment_count":
                len(group),
            "mean_quality_score":
                pd.to_numeric(
                    group[
                        "overall_quality_score"
                    ],
                    errors="coerce",
                ).mean(),
            "accept_rate":
                group[
                    "validator_decision"
                ].astype(str)
                .eq("ACCEPT")
                .mean(),
            "revise_rate":
                group[
                    "validator_decision"
                ].astype(str)
                .eq("REVISE")
                .mean(),
            "reject_rate":
                group[
                    "validator_decision"
                ].astype(str)
                .eq("REJECT")
                .mean(),
        }

        for col in SCORE_COLUMNS:
            row[
                f"mean_{col}"
            ] = pd.to_numeric(
                group[col],
                errors="coerce",
            ).mean()

        cross_rows.append(
            row
        )

    cross_matrix = pd.DataFrame(
        cross_rows
    )

    cross_matrix_path = (
        out_dir
        / "generator_by_judge_matrix.csv"
    )

    cross_matrix.to_csv(
        cross_matrix_path,
        index=False,
    )

    # ---------------------------------------------------------------
    # Pairwise judge agreement and score distance
    # ---------------------------------------------------------------
    pair_rows = []

    judges = sorted(
        judgments[
            "judge_family_id"
        ].astype(str).unique()
    )

    decision_pivot = (
        judgments.pivot_table(
            index="candidate_id",
            columns="judge_family_id",
            values="validator_decision",
            aggfunc="first",
        )
    )

    for a, b in combinations(
        judges,
        2,
    ):
        pair = (
            decision_pivot[
                [a, b]
            ]
            .dropna()
        )

        exact = (
            (
                pair[a].astype(str)
                == pair[b].astype(str)
            ).mean()
            if len(pair)
            else float("nan")
        )

        kappa = (
            _cohen_kappa(
                pair[a],
                pair[b],
            )
            if len(pair)
            else float("nan")
        )

        score_diffs = []

        for col in SCORE_COLUMNS:
            score_pivot = (
                judgments.pivot_table(
                    index="candidate_id",
                    columns="judge_family_id",
                    values=col,
                    aggfunc="first",
                )
            )

            if (
                a in score_pivot.columns
                and b in score_pivot.columns
            ):
                vals = (
                    score_pivot[
                        [a, b]
                    ]
                    .apply(
                        pd.to_numeric,
                        errors="coerce",
                    )
                    .dropna()
                )

                if not vals.empty:
                    score_diffs.extend(
                        (
                            vals[a]
                            - vals[b]
                        )
                        .abs()
                        .tolist()
                    )

        pair_rows.append({
            "judge_a":
                a,
            "judge_b":
                b,
            "candidate_count":
                len(pair),
            "exact_decision_agreement":
                exact,
            "cohen_kappa":
                kappa,
            "mean_absolute_score_difference":
                (
                    sum(score_diffs)
                    / len(score_diffs)
                    if score_diffs
                    else float("nan")
                ),
        })

    pairwise = pd.DataFrame(
        pair_rows
    )

    pairwise_path = (
        out_dir
        / "judge_pairwise_agreement.csv"
    )

    pairwise.to_csv(
        pairwise_path,
        index=False,
    )

    # ---------------------------------------------------------------
    # Text report
    # ---------------------------------------------------------------
    report_path = (
        out_dir
        / "comparison_report.txt"
    )

    lines = [
        (
            "ChemBreak V4 experiment: "
            f"{config['experiment_id']}"
        ),
        "",
        "IMPORTANT",
        (
            "The automated generator ranking "
            "uses cross-family judges."
        ),
        (
            "Judge accuracy cannot be "
            "established from model-model "
            "agreement alone."
        ),
        (
            "Complete the blinded human-review "
            "sample before selecting final judges."
        ),
        "",
        "GENERATOR SUMMARY",
        generator_summary.to_string(
            index=False
        ),
        "",
        "JUDGE BEHAVIOR SUMMARY",
        judge_summary.to_string(
            index=False
        ),
        "",
        "PAIRWISE JUDGE AGREEMENT",
        pairwise.to_string(
            index=False
        ),
    ]

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return {
        "candidate_consensus":
            candidate_consensus_path,
        "generator_summary":
            generator_summary_path,
        "judge_summary":
            judge_summary_path,
        "generator_by_judge_matrix":
            cross_matrix_path,
        "judge_pairwise_agreement":
            pairwise_path,
        "comparison_report":
            report_path,
    }
