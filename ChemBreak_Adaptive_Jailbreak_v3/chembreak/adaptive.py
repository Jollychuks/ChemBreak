from __future__ import annotations
from typing import Any
from .evaluator import evaluate
from .prompts import adaptive_next_prompt
from .schemas import adaptive_decision_schema


def _turn_log(cfg, message: str) -> None:
    if cfg.get("runtime", {}).get("show_turn_progress", True):
        print(message, flush=True)


def _attacker_log(cfg, message: str) -> None:
    runtime = cfg.get("runtime", {})
    if runtime.get("show_turn_progress", True) and runtime.get("show_attacker_progress", True):
        print(message, flush=True)


def _attacker_history(turn_records: list[dict[str, Any]], use_judge_feedback: bool) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for turn in turn_records:
        item: dict[str, Any] = {
            "user": turn["user"],
            "assistant": turn["assistant"],
            "route_id": turn.get("route_id", ""),
        }
        if use_judge_feedback and "evaluation" in turn:
            ev = turn["evaluation"]
            item["judge_feedback"] = {
                "alignment_breach": bool(ev.get("alignment_breach", False)),
                "effective_chemical_breach": bool(ev.get("effective_chemical_breach", False)),
                "material_disagreement": bool(ev.get("material_disagreement", False)),
            }
        history.append(item)
    return history


def _attacker_json(attacker: Any, prompt: str, schema: dict[str, Any]) -> Any:
    try:
        return attacker.json(prompt, schema=schema)
    except TypeError:
        return attacker.json(prompt)


def run_adaptive(task, asset, target: Any, attacker, judges, cfg, target_name: str = "target"):
    target.reset()
    graph = asset["graph"]
    current_route = asset["adaptive_start_route"]
    route_map = {r["route_id"]: r for r in graph["routes"]}
    budget = int(cfg["attack_budget"]["adaptive_queries"])
    max_switches = int(cfg["attack_budget"].get("max_route_switches", 2))
    use_judge_feedback = bool(cfg.get("adaptive", {}).get("attacker_uses_judge_feedback", False))
    switches = 0
    conversation: list[dict[str, str]] = []
    turn_records: list[dict[str, Any]] = []
    route_trace: list[str] = []
    success = False
    termination = "budget_exhausted"
    q = route_map[current_route]["first_query"]
    attacker_model = getattr(attacker, "model", cfg.get("models", {}).get("attacker", {}).get("model", "attacker"))
    _attacker_log(
        cfg,
        f"[C3 ATTACKER] model={attacker_model} | {target_name} | {task['task_id']} | "
        f"initial_route={current_route} | judge_feedback={'ON' if use_judge_feedback else 'OFF'}",
    )

    for n in range(1, budget + 1):
        route_trace.append(current_route)
        _turn_log(cfg, f"[C3] {target_name} | {task['task_id']} | query {n}/{budget} | route={current_route} | switches={switches}/{max_switches}")
        r = target.ask(q)
        conversation.append({"user": q, "assistant": r})
        ev = evaluate(task, list(conversation), r, *judges, cfg)
        turn_records.append({"user": q, "assistant": r, "route_id": current_route, "evaluation": ev})
        if ev["success"]:
            success = True
            termination = "validated_success"
            break
        remaining = budget - n
        if remaining <= 0:
            break

        visible_history = _attacker_history(turn_records, use_judge_feedback)
        _attacker_log(
            cfg,
            f"[C3 ATTACKER] model={attacker_model} | {target_name} | {task['task_id']} | "
            f"planning next query after turn={n} | current_route={current_route} | remaining={remaining}",
        )
        decision = _attacker_json(
            attacker,
            adaptive_next_prompt(task, graph, visible_history, current_route, remaining, switches, max_switches),
            adaptive_decision_schema(list(route_map.keys())),
        )
        action = str(decision.get("action", "REFINE")).upper()
        selected = str(decision.get("selected_route_id", current_route))
        previous_route = current_route
        if action == "STOP_FAILURE":
            termination = "attacker_stop_failure"
            _attacker_log(cfg, f"[C3 ATTACKER] {task['task_id']} | action=STOP_FAILURE | route={current_route}")
            break
        if action == "SWITCH":
            if switches >= max_switches or selected not in route_map or selected == current_route:
                action = "REFINE"
                selected = current_route
            else:
                switches += 1
                current_route = selected
        elif selected in route_map:
            current_route = selected

        _attacker_log(
            cfg,
            f"[C3 ATTACKER] {task['task_id']} | action={action} | route={previous_route}->{current_route} | "
            f"switches={switches}/{max_switches}",
        )
        q = str(decision.get("next_query", "")).strip()
        if not q:
            termination = "empty_next_query"
            break

    return {
        "condition": "C3_adaptive_chembreak",
        "queries_used": len(turn_records),
        "route_switches": switches,
        "route_trace": route_trace,
        "attacker_model": attacker_model,
        "attacker_uses_judge_feedback": use_judge_feedback,
        "turns": turn_records,
        "success": success,
        "termination_reason": termination,
    }
