from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from .utils import utc_now, write_json


def _json(value):
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value or "{}")
    except Exception:
        return {}


def _metric(judgment, name):
    value = _json(judgment).get(str(name).lower())
    return value if isinstance(value, dict) else {}


def export_results(store, output_dir, target_id, config):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    private = output / "PRIVATE_AUDIT"
    private.mkdir(parents=True, exist_ok=True)

    episodes = pd.DataFrame(store.episodes())
    turns = pd.DataFrame(store.turns())
    queries = pd.DataFrame(store.target_queries())
    events = pd.DataFrame(store.provider_events())
    calls = pd.DataFrame(store.role_calls())
    turn_metadata = {
        int(row["target_query_index"]): _json(row.get("decision_json"))
        for row in store.turns()
        if row.get("target_query_index") is not None
    }
    if not turns.empty:
        turns["strategy_mode"] = turns["decision_json"].map(
            lambda value: str(_json(value).get("strategy_mode", "baseline"))
        )
        turns["progressive_attempt"] = turns["decision_json"].map(
            lambda value: int(_json(value).get("progressive_attempt", 0) or 0)
        )
        turns["buildup_level"] = turns["decision_json"].map(
            lambda value: _json(value).get("buildup_level")
        )
        turns["buildup_route"] = turns["decision_json"].map(
            lambda value: str(_json(value).get("buildup_route", ""))
        )
        turns["trajectory_role"] = turns["decision_json"].map(
            lambda value: str(_json(value).get("trajectory_role", ""))
        )
    if not episodes.empty:
        episodes["strategy_mode"] = episodes.apply(
            lambda row: (
                "baseline" if row["phase"] == "baseline" else
                "adaptive_actions" if int(row["stage"]) == 1 else "progressive_buildup"
            ),
            axis=1,
        )
        episodes["progressive_attempt"] = episodes["stage"].map(
            lambda stage: 1 if int(stage) == 2 else (2 if int(stage) == 3 else 0)
        )

    task_ids = [str(item) for item in store.get_meta("experiment_identity", {}).get("assignment_ids", [])]
    task_rows = []
    score_rows = []
    transcript_rows = []
    for query in store.target_queries():
        decision = turn_metadata.get(int(query["query_index"]), {})
        judgment = _json(query.get("judgment_json"))
        rhs = _metric(judgment, "rhs")
        ofs = _metric(judgment, "ofs")
        transcript_rows.append(
            {
                "query_index": int(query["query_index"]),
                "assignment_id": query["assignment_id"],
                "target": target_id,
                "phase": query["phase"],
                "stage": int(query["stage"]),
                "turn": int(query["turn_index"]),
                "context_id": query["context_id"],
                "action": query["action_id"],
                "strategy_mode": decision.get("strategy_mode", "baseline"),
                "progressive_attempt": int(decision.get("progressive_attempt", 0) or 0),
                "buildup_level": decision.get("buildup_level"),
                "buildup_route": decision.get("buildup_route", ""),
                "trajectory_role": decision.get("trajectory_role", ""),
                "prompt": query["prompt"],
                "response": query["response"],
                "judgment_status": query["judgment_status"],
                "rhs": rhs.get("score"),
                "rhs_effective": rhs.get("effective_score"),
                "rhs_status": rhs.get("decision_status"),
                "rhs_evidence_status": rhs.get("evidence_status"),
                "ofs": ofs.get("score"),
                "ofs_effective": ofs.get("effective_score"),
                "ofs_status": ofs.get("decision_status"),
                "ofs_evidence_status": ofs.get("evidence_status"),
                "terminal_ofs5": bool(judgment.get("terminal_ofs5", False)),
                "high_rhs_low_ofs": bool(judgment.get("high_rhs_low_ofs", False)),
                "calibration_flags_json": json.dumps(
                    judgment.get("calibration_flags", []), ensure_ascii=False
                ),
                "created_at": query["created_at"],
                "judged_at": query["judged_at"],
            }
        )
        for name, result in (("RHS", rhs), ("OFS", ofs)):
            score_rows.append(
                {
                    "query_index": int(query["query_index"]),
                    "assignment_id": query["assignment_id"],
                    "target": target_id,
                    "phase": query["phase"],
                    "stage": int(query["stage"]),
                    "turn": int(query["turn_index"]),
                    "context_id": query["context_id"],
                    "strategy_mode": decision.get("strategy_mode", "baseline"),
                    "progressive_attempt": int(decision.get("progressive_attempt", 0) or 0),
                    "buildup_level": decision.get("buildup_level"),
                    "metric": name,
                    "score": result.get("score"),
                    "effective_score": result.get("effective_score"),
                    "score_status": result.get("score_status"),
                    "evidence_status": result.get("evidence_status"),
                    "decision_status": result.get("decision_status", "unresolved"),
                    "publication_ready": result.get("publication_ready", False),
                    "confidence": result.get("confidence"),
                    "reason_code": result.get("reason_code"),
                    "scientific_credibility": result.get("scientific_credibility"),
                    "validation_errors_json": json.dumps(result.get("validation_errors", []), ensure_ascii=False),
                    "error_category": (result.get("error") or {}).get("category"),
                    "error_code": (result.get("error") or {}).get("error_code"),
                    "error_status": (result.get("error") or {}).get("status"),
                }
            )

    transcript = pd.DataFrame(transcript_rows)
    scores = pd.DataFrame(score_rows)
    for assignment_id in task_ids:
        task_scores = scores[scores.assignment_id == assignment_id] if not scores.empty else pd.DataFrame()
        task_transcript = transcript[transcript.assignment_id == assignment_id] if not transcript.empty else pd.DataFrame()
        ofs5 = task_transcript[task_transcript.terminal_ofs5 == True] if not task_transcript.empty else pd.DataFrame()
        first = ofs5.sort_values("query_index").iloc[0].to_dict() if not ofs5.empty else None
        baseline_ofs5 = ofs5[ofs5.phase == "baseline"] if not ofs5.empty else pd.DataFrame()
        adaptive_ofs5 = ofs5[ofs5.phase == "adaptive"] if not ofs5.empty else pd.DataFrame()
        first_adaptive = (
            adaptive_ofs5.sort_values("query_index").iloc[0].to_dict()
            if not adaptive_ofs5.empty else None
        )
        rhs_values = task_scores[(task_scores.metric == "RHS") & task_scores.effective_score.notna()].effective_score.tolist() if not task_scores.empty else []
        ofs_values = task_scores[(task_scores.metric == "OFS") & task_scores.effective_score.notna()].effective_score.tolist() if not task_scores.empty else []
        baseline_scores = task_scores[task_scores.phase == "baseline"] if not task_scores.empty else pd.DataFrame()
        adaptive_scores = task_scores[task_scores.phase == "adaptive"] if not task_scores.empty else pd.DataFrame()
        baseline_rhs = baseline_scores[(baseline_scores.metric == "RHS") & baseline_scores.effective_score.notna()].effective_score.tolist() if not baseline_scores.empty else []
        baseline_ofs = baseline_scores[(baseline_scores.metric == "OFS") & baseline_scores.effective_score.notna()].effective_score.tolist() if not baseline_scores.empty else []
        adaptive_rhs = adaptive_scores[(adaptive_scores.metric == "RHS") & adaptive_scores.effective_score.notna()].effective_score.tolist() if not adaptive_scores.empty else []
        adaptive_ofs = adaptive_scores[(adaptive_scores.metric == "OFS") & adaptive_scores.effective_score.notna()].effective_score.tolist() if not adaptive_scores.empty else []
        task_rows.append(
            {
                "assignment_id": assignment_id,
                "target": target_id,
                "ofs_5_reached": first is not None,
                "baseline_ofs_5_reached": not baseline_ofs5.empty,
                "adaptive_ofs_5_reached": not adaptive_ofs5.empty,
                "first_ofs_5_phase": first.get("phase") if first else None,
                "first_ofs_5_stage": first.get("stage") if first else None,
                "first_ofs_5_turn": first.get("turn") if first else None,
                "adaptive_first_ofs_5_stage": first_adaptive.get("stage") if first_adaptive else None,
                "adaptive_first_ofs_5_turn": first_adaptive.get("turn") if first_adaptive else None,
                "peak_rhs": max(rhs_values) if rhs_values else None,
                "peak_ofs": max(ofs_values) if ofs_values else None,
                "baseline_rhs": baseline_rhs[0] if baseline_rhs else None,
                "baseline_ofs": baseline_ofs[0] if baseline_ofs else None,
                "adaptive_peak_rhs": max(adaptive_rhs) if adaptive_rhs else None,
                "adaptive_peak_ofs": max(adaptive_ofs) if adaptive_ofs else None,
                "unresolved_rhs": int(((task_scores.metric == "RHS") & (task_scores.decision_status != "valid")).sum()) if not task_scores.empty else 0,
                "unresolved_ofs": int(((task_scores.metric == "OFS") & (task_scores.decision_status != "valid")).sum()) if not task_scores.empty else 0,
                "target_queries": len(task_transcript),
                "high_rhs_low_ofs_judgments": int(
                    task_transcript.high_rhs_low_ofs.sum()
                ) if not task_transcript.empty else 0,
            }
        )

    tasks = pd.DataFrame(task_rows)
    unresolved_rhs = int(((scores.metric == "RHS") & (scores.decision_status != "valid")).sum()) if not scores.empty else 0
    unresolved_ofs = int(((scores.metric == "OFS") & (scores.decision_status != "valid")).sum()) if not scores.empty else 0
    failure_distribution = Counter(
        (str(row.get("metric") or "NONE"), str(row.get("error_category") or row.get("event_type") or "unknown"))
        for row in store.provider_events()
        if row.get("event_type") in {"judge_error", "policy_block", "target_error", "planner_error", "gate_error"}
    )
    summary = {
        "project": "ChemBreak",
        "generated_at_utc": utc_now(),
        "target": target_id,
        "judge_experiment": config["run"]["judge_experiment"],
        "judge_model": config["roles"]["judge_llm"]["model"],
        "scheduled_tasks": len(task_ids),
        "completed_tasks": int(len(tasks)),
        "ofs_5_tasks": int(tasks.ofs_5_reached.sum()) if not tasks.empty else 0,
        "baseline_ofs_5_tasks": int(tasks.baseline_ofs_5_reached.sum()) if not tasks.empty else 0,
        "adaptive_ofs_5_tasks": int(tasks.adaptive_ofs_5_reached.sum()) if not tasks.empty else 0,
        "target_queries": int(len(queries)),
        "adaptive_stages": int(config["experiment"]["adaptive_stages"]),
        "turns_per_stage": int(config["experiment"]["max_turns_per_stage"]),
        "adaptive_target_queries_by_stage": {
            str(stage): int(((queries.phase == "adaptive") & (queries.stage == stage)).sum())
            for stage in (1, 2, 3)
        } if not queries.empty else {"1": 0, "2": 0, "3": 0},
        "adaptive_ofs_5_tasks_by_stage": {
            str(stage): int(tasks.adaptive_first_ofs_5_stage.eq(stage).sum())
            for stage in (1, 2, 3)
        } if not tasks.empty else {"1": 0, "2": 0, "3": 0},
        "unresolved_rhs": unresolved_rhs,
        "unresolved_ofs": unresolved_ofs,
        "rhs_score_distribution": dict(Counter(str(int(value)) for value in scores[(scores.metric == "RHS") & scores.score.notna()].score)) if not scores.empty else {},
        "ofs_score_distribution": dict(Counter(str(int(value)) for value in scores[(scores.metric == "OFS") & scores.score.notna()].score)) if not scores.empty else {},
        "baseline_rhs_score_distribution": dict(Counter(str(int(value)) for value in scores[(scores.phase == "baseline") & (scores.metric == "RHS") & scores.score.notna()].score)) if not scores.empty else {},
        "baseline_ofs_score_distribution": dict(Counter(str(int(value)) for value in scores[(scores.phase == "baseline") & (scores.metric == "OFS") & scores.score.notna()].score)) if not scores.empty else {},
        "adaptive_rhs_score_distribution": dict(Counter(str(int(value)) for value in scores[(scores.phase == "adaptive") & (scores.metric == "RHS") & scores.score.notna()].score)) if not scores.empty else {},
        "adaptive_ofs_score_distribution": dict(Counter(str(int(value)) for value in scores[(scores.phase == "adaptive") & (scores.metric == "OFS") & scores.score.notna()].score)) if not scores.empty else {},
        "high_rhs_low_ofs_judgments": int(
            transcript.high_rhs_low_ofs.sum()
        ) if not transcript.empty else 0,
        "failure_distribution": {
            f"{metric}:{category}": count for (metric, category), count in sorted(failure_distribution.items())
        },
        "measurement": {
            "rhs_scale": "1-5",
            "ofs_scale": "1-5",
            "scope": "cumulative target responses in the active conversation",
            "context_restart_resets_evidence": True,
            "metrics_called_independently": True,
            "ofs_drives_reward_state_and_planner": True,
            "rhs_observational_only": True,
            "sole_judge_model": config["roles"]["judge_llm"]["model"],
        },
    }

    tasks.to_csv(output / "tasks.csv", index=False)
    scores.to_csv(output / "scores.csv", index=False)
    episodes.to_csv(output / "stages.csv", index=False)
    events.to_csv(output / "provider_events.csv", index=False)
    calls.to_csv(output / "role_calls.csv", index=False)

    public_turns = turns.copy()
    if not public_turns.empty:
        public_turns["prompt"] = "[REDACTED]"
        public_turns["response"] = "[REDACTED]"
    public_turns.to_csv(output / "turns.csv", index=False)
    public_queries = queries.copy()
    if not public_queries.empty:
        public_queries["prompt"] = "[REDACTED]"
        public_queries["response"] = "[REDACTED]"
    public_queries.to_csv(output / "target_queries.csv", index=False)

    transcript.to_csv(private / "full_transcripts.csv", index=False)
    transcript.to_json(private / "full_transcripts.jsonl", orient="records", lines=True, force_ascii=False)
    write_json(output / "summary.json", summary)
    write_json(output / "experiment_identity.json", store.get_meta("experiment_identity", {}))
    return summary
