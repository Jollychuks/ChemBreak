from __future__ import annotations

import json
import sqlite3
from itertools import pairwise
from pathlib import Path

import pandas as pd


def _json_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([json.loads(value) for value in frame.get("record_json", [])])


def _load(db_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(db_path) as connection:
        episodes = pd.read_sql_query("SELECT * FROM episodes", connection)
        transcripts = _json_frame(pd.read_sql_query("SELECT record_json FROM transcripts", connection))
        observations = _json_frame(pd.read_sql_query("SELECT record_json FROM observations", connection))
        evaluations = _json_frame(pd.read_sql_query("SELECT record_json FROM evaluations", connection))
        api_raw = pd.read_sql_query("SELECT episode_id,record_json,created_at FROM api_calls", connection)
        failures = pd.read_sql_query("SELECT * FROM failures", connection)
    api_rows = []
    for row in api_raw.to_dict(orient="records"):
        record = json.loads(row.pop("record_json"))
        usage = record.pop("usage", {})
        api_rows.append({**row, **record, "usage_json": json.dumps(usage, sort_keys=True)})
    return episodes, transcripts, observations, evaluations, pd.DataFrame(api_rows), failures


def _turns(transcripts: pd.DataFrame, observations: pd.DataFrame, evaluations: pd.DataFrame) -> pd.DataFrame:
    if transcripts.empty:
        return pd.DataFrame()
    turns = transcripts.copy()
    if not observations.empty:
        turns = turns.merge(observations, on=["episode_id", "turn_index"], how="left", suffixes=("", "_observer"))
    if not evaluations.empty:
        turns = turns.merge(evaluations, on=["episode_id", "turn_index"], how="left", suffixes=("", "_evaluation"))
    return turns


def _episode_results(episodes: pd.DataFrame, evaluations: pd.DataFrame) -> pd.DataFrame:
    records = []
    for episode in episodes.to_dict(orient="records"):
        rows = (
            evaluations[evaluations.episode_id == episode["episode_id"]].sort_values("turn_index")
            if not evaluations.empty else pd.DataFrame()
        )
        successful_turn = None
        if not rows.empty and "verified_success" in rows:
            successful = rows[rows.verified_success == True]
            if not successful.empty:
                successful_turn = int(successful.iloc[0].turn_index)
        complete = episode["status"] == "complete"
        success = bool(episode["success"]) if complete else None
        last = rows.iloc[-1].to_dict() if not rows.empty else {}
        records.append({
            "episode_id": episode["episode_id"], "assignment_id": episode["assignment_id"],
            "target_id": episode["target_id"], "condition": episode["condition"],
            "status": episode["status"], "evaluated": complete,
            "queries_used": int(episode["queries_used"]), "verified_success": success,
            "success_label": "YES" if success else ("NO" if complete else "NOT_EVALUATED"),
            "successful_turn": successful_turn,
            "bootstrap_success": bool(success and successful_turn == 1),
            "adaptive_success": bool(success and successful_turn is not None and successful_turn >= 2),
            "adaptive_opportunity": bool(complete and int(episode["queries_used"]) >= 2),
            "adaptive_steps_used": max(0, int(episode["queries_used"]) - 1),
            "terminal_reason": episode["terminal_reason"],
            "final_response_class": last.get("response_class", "not_evaluated"),
            "final_verification_status": last.get("verification_status", "not_evaluated"),
            "cumulative_reward": episode["cumulative_reward"],
        })
    return pd.DataFrame(records)


def _adaptive_metrics(results: pd.DataFrame) -> pd.DataFrame:
    completed = results[results.status == "complete"] if not results.empty else results
    records = []
    groups = list(completed.groupby("target_id")) if not completed.empty else []
    if not completed.empty:
        groups.append(("ALL_TARGETS", completed))
    for target, group in groups:
        successes = int(group.verified_success.sum())
        bootstrap = int(group.bootstrap_success.sum())
        adaptive = int(group.adaptive_success.sum())
        bootstrap_failures = len(group) - bootstrap
        records.append({
            "target_id": target, "completed_episodes": len(group),
            "verified_successes": successes,
            "overall_asr": successes / len(group) if len(group) else float("nan"),
            "bootstrap_successes": bootstrap,
            "bootstrap_asr": bootstrap / len(group) if len(group) else float("nan"),
            "bootstrap_failures": bootstrap_failures,
            "post_feedback_adaptive_successes": adaptive,
            "conditional_adaptive_asr": adaptive / bootstrap_failures if bootstrap_failures else float("nan"),
            "mean_queries_used": group.queries_used.mean(),
            "mean_adaptive_steps_used": group.adaptive_steps_used.mean(),
        })
    return pd.DataFrame(records)


def _asr_at_k(results: pd.DataFrame, evaluations: pd.DataFrame, budget: int) -> pd.DataFrame:
    completed = results[results.status == "complete"] if not results.empty else results
    records = []
    for target, group in completed.groupby("target_id"):
        ids = set(group.episode_id)
        relevant = evaluations[evaluations.episode_id.isin(ids)] if not evaluations.empty else pd.DataFrame()
        for k in range(1, budget + 1):
            wins = len(set(relevant.loc[(relevant.turn_index <= k) & (relevant.verified_success == True), "episode_id"])) if not relevant.empty else 0
            records.append({
                "target_id": target, "query_budget_k": k, "completed_episodes": len(group),
                "successes_at_k": wins, "asr_at_k": wins / len(group) if len(group) else float("nan"),
            })
    return pd.DataFrame(records)


def _action_metrics(turns: pd.DataFrame) -> pd.DataFrame:
    if turns.empty or "action_id" not in turns:
        return pd.DataFrame(columns=["target_id", "action_id", "uses", "verified_successes", "mean_reward"])
    adaptive = turns[turns.action_id != "BOOTSTRAP_DIRECT"].copy()
    if adaptive.empty:
        return pd.DataFrame(columns=["target_id", "action_id", "uses", "verified_successes", "mean_reward"])
    return adaptive.groupby(["target_id", "action_id"], dropna=False).agg(
        uses=("episode_id", "size"),
        verified_successes=("verified_success", "sum"),
        mean_reward=("reward", "mean"),
        mean_progress=("progress_score", "mean"),
    ).reset_index()


def _transitions(turns: pd.DataFrame) -> pd.DataFrame:
    records = []
    if turns.empty or "response_class" not in turns:
        return pd.DataFrame(columns=["target_id", "from_class", "action_id", "to_class", "count"])
    for episode_id, group in turns.sort_values(["episode_id", "turn_index"]).groupby("episode_id"):
        rows = group.to_dict(orient="records")
        for previous, current in pairwise(rows):
            records.append({
                "episode_id": episode_id, "target_id": current["target_id"],
                "from_class": previous.get("response_class"), "action_id": current.get("action_id"),
                "to_class": current.get("response_class"),
            })
    if not records:
        return pd.DataFrame(columns=["target_id", "from_class", "action_id", "to_class", "count"])
    return pd.DataFrame(records).groupby(
        ["target_id", "from_class", "action_id", "to_class"], dropna=False
    ).size().reset_index(name="count")


def export_results(
    db_path: str | Path, selection_path: str | Path, output_dir: str | Path,
    query_budget: int, release_raw_outputs: bool,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    private_dir = output_dir / "private"
    release_dir = output_dir / "release"
    private_dir.mkdir(parents=True, exist_ok=True)
    release_dir.mkdir(parents=True, exist_ok=True)
    episodes, transcripts, observations, evaluations, api_calls, failures = _load(Path(db_path))
    turns = _turns(transcripts, observations, evaluations)
    results = _episode_results(episodes, evaluations)
    adaptive = _adaptive_metrics(results)

    results.to_csv(release_dir / "episode_results.csv", index=False)
    adaptive.to_csv(release_dir / "adaptive_mdp_metrics.csv", index=False)
    _asr_at_k(results, evaluations, query_budget).to_csv(release_dir / "asr_by_query_budget.csv", index=False)
    _action_metrics(turns).to_csv(release_dir / "action_metrics.csv", index=False)
    _transitions(turns).to_csv(release_dir / "state_action_transitions.csv", index=False)
    coverage = episodes.groupby(["target_id", "status"]).size().reset_index(name="episodes") if not episodes.empty else pd.DataFrame()
    coverage.to_csv(release_dir / "run_coverage.csv", index=False)
    failures.to_csv(release_dir / "failures.csv", index=False)
    pd.read_csv(selection_path).to_csv(release_dir / "selected_tasks.csv", index=False)
    if not api_calls.empty:
        api_calls.groupby(["role", "status"], dropna=False).size().reset_index(name="calls").to_csv(
            release_dir / "role_call_counts.csv", index=False
        )
    else:
        pd.DataFrame(columns=["role", "status", "calls"]).to_csv(release_dir / "role_call_counts.csv", index=False)

    api_calls.to_csv(private_dir / "api_calls.csv", index=False)
    for name, frame in (("transcripts_raw", transcripts), ("observations", observations), ("evaluations", evaluations)):
        with (private_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for record in frame.to_dict(orient="records"):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    redacted = transcripts.copy()
    for column in ("attack_prompt", "target_response"):
        if column in redacted:
            redacted[column] = "[REDACTED FROM RELEASE EXPORT]"
    redacted.to_csv(release_dir / "transcripts_redacted.csv", index=False)
    observations.to_csv(release_dir / "observations.csv", index=False)
    evaluations.to_csv(release_dir / "evaluations.csv", index=False)
    successful_ids = set(results.loc[results.verified_success == True, "episode_id"].astype(str)) if not results.empty else set()
    successful_turns = turns[turns.episode_id.astype(str).isin(successful_ids)] if not turns.empty else turns
    successful_turns.to_csv(private_dir / "successful_attacks_private.csv", index=False)
    if release_raw_outputs:
        (release_dir / "transcripts_raw.jsonl").write_bytes((private_dir / "transcripts_raw.jsonl").read_bytes())

    manifest = {
        "release_directory": str(release_dir),
        "episode_results": str(release_dir / "episode_results.csv"),
        "adaptive_mdp_metrics": str(release_dir / "adaptive_mdp_metrics.csv"),
        "action_metrics": str(release_dir / "action_metrics.csv"),
        "state_action_transitions": str(release_dir / "state_action_transitions.csv"),
        "private_successful_attacks": str(private_dir / "successful_attacks_private.csv"),
    }
    (output_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
