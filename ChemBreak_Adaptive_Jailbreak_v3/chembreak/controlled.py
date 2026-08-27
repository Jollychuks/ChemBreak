from __future__ import annotations
from typing import Any
from .evaluator import evaluate


def _turn_log(cfg, message: str) -> None:
    if cfg.get("runtime", {}).get("show_turn_progress", True):
        print(message, flush=True)


def direct_single(task, target: Any, judges, cfg, target_name: str = "target"):
    target.reset()
    q = task["benchmark_prompt"]
    _turn_log(cfg, f"[C0] {target_name} | {task['task_id']} | query 1/1")
    r = target.ask(q)
    conversation = [{"user": q, "assistant": r}]
    ev = evaluate(task, conversation, r, *judges, cfg)
    return {
        "condition": "C0_direct_single",
        "queries_used": 1,
        "turns": [{"user": q, "assistant": r, "evaluation": ev}],
        "success": ev["success"],
    }


def repeated_single(task, asset, target: Any, judges, cfg, target_name: str = "target"):
    attempts = []
    success = False
    total = len(asset["repeated_single"])
    for i, item in enumerate(asset["repeated_single"], 1):
        target.reset()
        q = item["query"]
        _turn_log(cfg, f"[C1] {target_name} | {task['task_id']} | independent attempt {i}/{total}")
        r = target.ask(q)
        conversation = [{"user": q, "assistant": r}]
        ev = evaluate(task, conversation, r, *judges, cfg)
        attempts.append({"attempt": i, "route_family": item.get("route_family"), "query": q, "response": r, "evaluation": ev})
        if ev["success"]:
            success = True
            break
    return {"condition": "C1_repeated_single", "queries_used": len(attempts), "attempts": attempts, "success": success}


def fixed_multi(task, asset, target: Any, judges, cfg, target_name: str = "target"):
    target.reset()
    conversation: list[dict[str, str]] = []
    turn_records: list[dict[str, Any]] = []
    success = False
    queries = asset["fixed_multi"]["queries"]
    for i, q in enumerate(queries, 1):
        _turn_log(cfg, f"[C2] {target_name} | {task['task_id']} | fixed turn {i}/{len(queries)}")
        r = target.ask(q)
        conversation.append({"user": q, "assistant": r})
        # Judge sees only target conversation content. Earlier judge outputs never re-enter judge context.
        ev = evaluate(task, list(conversation), r, *judges, cfg)
        turn_records.append({"user": q, "assistant": r, "evaluation": ev})
        if ev["success"]:
            success = True
            break
    return {
        "condition": "C2_fixed_multi",
        "route_family": asset["fixed_multi"].get("route_family"),
        "queries_used": len(turn_records),
        "turns": turn_records,
        "success": success,
    }
