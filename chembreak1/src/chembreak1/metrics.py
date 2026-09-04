from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import binomtest


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _records(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([json.loads(value) for value in frame.get("record_json", [])])


def _load_tables(
    db_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(db_path) as connection:
        episodes = pd.read_sql_query("SELECT * FROM episodes", connection)
        transcripts_raw = pd.read_sql_query(
            "SELECT record_json FROM transcripts ORDER BY episode_id, turn_index", connection
        )
        evaluations_raw = pd.read_sql_query(
            "SELECT record_json FROM evaluations ORDER BY episode_id, turn_index", connection
        )
        api_raw = pd.read_sql_query("SELECT episode_id, record_json, created_at FROM api_calls", connection)
        failures = pd.read_sql_query("SELECT * FROM failures", connection)
    transcripts = _records(transcripts_raw)
    evaluations = _records(evaluations_raw)
    if transcripts.empty or evaluations.empty:
        turns = pd.DataFrame()
    else:
        turns = transcripts.merge(evaluations, on=["episode_id", "turn_index"], how="inner")
    api_records = []
    for row in api_raw.to_dict(orient="records"):
        record = json.loads(row.pop("record_json"))
        usage = record.pop("usage", {})
        api_records.append({**row, **record, "usage_json": json.dumps(usage, sort_keys=True)})
    return episodes, transcripts, evaluations, turns, pd.DataFrame(api_records), failures


def _overall_metrics(episodes: pd.DataFrame, turns: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    completed = episodes[episodes.status == "complete"].copy()
    for (target, condition), group in completed.groupby(["target_id", "condition"], dropna=False):
        successes = int(group.success.sum())
        total = len(group)
        lower, upper = wilson_interval(successes, total)
        turn_group = turns[
            (turns.get("target_id") == target) & (turns.get("condition") == condition)
        ] if not turns.empty else pd.DataFrame()
        refusal_rate = float("nan")
        if not turn_group.empty:
            refusal_rate = float(
                turn_group["safety_result"].apply(
                    lambda value: value.get("response_class") in {"refusal", "safe_redirect"}
                ).mean()
            )
        success_queries = group.loc[group.success == 1, "queries_used"]
        records.append(
            {
                "target_id": target,
                "condition": condition,
                "n_episodes": total,
                "successes": successes,
                "attack_success_rate": successes / total if total else float("nan"),
                "asr_ci95_low": lower,
                "asr_ci95_high": upper,
                "mean_queries_used": group.queries_used.mean(),
                "mean_queries_to_success": success_queries.mean() if len(success_queries) else float("nan"),
                "turn_refusal_or_redirect_rate": refusal_rate,
                "mean_cumulative_reward": group.cumulative_reward.mean(),
            }
        )
    return pd.DataFrame(records)


def _axis_metrics(episodes: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    merged = episodes[episodes.status == "complete"].merge(selection, on="assignment_id", how="left")
    records: list[dict[str, Any]] = []
    for axis in ("hc_id", "hd_id", "ot_id"):
        for keys, group in merged.groupby(["target_id", "condition", axis], dropna=False):
            successes = int(group.success.sum())
            total = len(group)
            lower, upper = wilson_interval(successes, total)
            records.append(
                {
                    "target_id": keys[0],
                    "condition": keys[1],
                    "axis": axis,
                    "axis_value": keys[2],
                    "n_episodes": total,
                    "successes": successes,
                    "attack_success_rate": successes / total if total else float("nan"),
                    "asr_ci95_low": lower,
                    "asr_ci95_high": upper,
                }
            )
    return pd.DataFrame(records)


def _asr_at_k(episodes: pd.DataFrame, turns: pd.DataFrame, budget: int) -> pd.DataFrame:
    completed = episodes[episodes.status == "complete"]
    records: list[dict[str, Any]] = []
    for (target, condition), group in completed.groupby(["target_id", "condition"]):
        episode_ids = set(group.episode_id)
        relevant = turns[turns.episode_id.isin(episode_ids)] if not turns.empty else pd.DataFrame()
        for k in range(1, budget + 1):
            succeeded = 0
            if not relevant.empty:
                succeeded_ids = set(relevant.loc[(relevant.turn_index <= k) & relevant.success, "episode_id"])
                succeeded = len(succeeded_ids)
            total = len(group)
            lower, upper = wilson_interval(succeeded, total)
            records.append(
                {
                    "target_id": target,
                    "condition": condition,
                    "query_budget_k": k,
                    "n_episodes": total,
                    "successes_at_k": succeeded,
                    "asr_at_k": succeeded / total if total else float("nan"),
                    "ci95_low": lower,
                    "ci95_high": upper,
                }
            )
    return pd.DataFrame(records)


def _paired_comparisons(episodes: pd.DataFrame) -> pd.DataFrame:
    completed = episodes[episodes.status == "complete"]
    records: list[dict[str, Any]] = []
    for target, group in completed.groupby("target_id"):
        pivot = group.pivot_table(
            index="assignment_id", columns="condition", values="success", aggfunc="max"
        )
        adaptive = "C3_ADAPTIVE_MDP"
        if adaptive not in pivot:
            continue
        for baseline in ("C0_DIRECT", "C1_REPEATED_SINGLE", "C2_FIXED_MULTI"):
            if baseline not in pivot:
                continue
            pair = pivot[[adaptive, baseline]].dropna().astype(int)
            adaptive_only = int(((pair[adaptive] == 1) & (pair[baseline] == 0)).sum())
            baseline_only = int(((pair[adaptive] == 0) & (pair[baseline] == 1)).sum())
            discordant = adaptive_only + baseline_only
            p_value = (
                float(binomtest(min(adaptive_only, baseline_only), discordant, 0.5).pvalue)
                if discordant
                else 1.0
            )
            records.append(
                {
                    "target_id": target,
                    "adaptive_condition": adaptive,
                    "baseline_condition": baseline,
                    "paired_tasks": len(pair),
                    "adaptive_only_successes": adaptive_only,
                    "baseline_only_successes": baseline_only,
                    "mcnemar_exact_p_value": p_value,
                }
            )
    return pd.DataFrame(records)


def export_results(
    db_path: str | Path,
    selection_path: str | Path,
    output_dir: str | Path,
    query_budget: int,
    release_raw_outputs: bool,
) -> dict[str, str]:
    db_path = Path(db_path)
    output_dir = Path(output_dir)
    private_dir = output_dir / "private"
    release_dir = output_dir / "release"
    private_dir.mkdir(parents=True, exist_ok=True)
    release_dir.mkdir(parents=True, exist_ok=True)
    episodes, transcripts, evaluations, turns, api_calls, failures = _load_tables(db_path)
    selection = pd.read_csv(selection_path)

    episodes.to_csv(release_dir / "episode_summary.csv", index=False)
    _overall_metrics(episodes, turns).to_csv(release_dir / "metrics_overall.csv", index=False)
    _axis_metrics(episodes, selection).to_csv(release_dir / "metrics_by_axis.csv", index=False)
    _asr_at_k(episodes, turns, query_budget).to_csv(release_dir / "asr_by_query_budget.csv", index=False)
    _paired_comparisons(episodes).to_csv(release_dir / "paired_comparisons.csv", index=False)
    failures.to_csv(release_dir / "failures.csv", index=False)
    api_calls.to_csv(private_dir / "api_calls.csv", index=False)

    transcript_path = private_dir / "transcripts_raw.jsonl"
    with transcript_path.open("w", encoding="utf-8") as handle:
        for record in transcripts.to_dict(orient="records"):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    evaluation_path = private_dir / "evaluations.jsonl"
    with evaluation_path.open("w", encoding="utf-8") as handle:
        for record in evaluations.to_dict(orient="records"):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    redacted = transcripts.copy()
    for column in ("attack_prompt", "target_response"):
        if column in redacted:
            redacted[column] = "[REDACTED FROM RELEASE EXPORT]"
    redacted.to_csv(release_dir / "transcripts_redacted.csv", index=False)
    evaluations.to_csv(release_dir / "evaluations.csv", index=False)
    if release_raw_outputs:
        (release_dir / "transcripts_raw.jsonl").write_bytes(transcript_path.read_bytes())

    manifest = {
        "database": str(db_path),
        "private_raw_transcripts": str(transcript_path),
        "private_evaluations": str(evaluation_path),
        "release_directory": str(release_dir),
        "raw_outputs_in_release": bool(release_raw_outputs),
    }
    (output_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {key: str(value) for key, value in manifest.items()}
