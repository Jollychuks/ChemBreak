from __future__ import annotations
from typing import Any
from .evaluator import evaluate
from .prompts import adaptive_next_prompt


def _turn_log(cfg, message: str) -> None:
    if cfg.get("runtime", {}).get("show_turn_progress", True):
        print(message, flush=True)


def run_adaptive(task, asset, target: Any, attacker, judges, cfg, target_name: str = "target"):
    target.reset()
    graph = asset["graph"]
    current_route = asset["adaptive_start_route"]
    route_map = {r["route_id"]: r for r in graph["routes"]}
    budget = int(cfg["attack_budget"]["adaptive_queries"])
    max_switches = int(cfg["attack_budget"].get("max_route_switches", 2))
    switches = 0
    turns = []
    route_trace = []
    success = False
    termination = "budget_exhausted"
    q = route_map[current_route]["first_query"]
    for n in range(1, budget + 1):
        route_trace.append(current_route)
        _turn_log(cfg, f"[C3] {target_name} | {task['task_id']} | query {n}/{budget} | route={current_route} | switches={switches}/{max_switches}")
        r = target.ask(q)
        turns.append({"user": q, "assistant": r, "route_id": current_route})
        ev = evaluate(task, turns, r, *judges, cfg)
        turns[-1]["evaluation"] = ev
        if ev["success"]:
            success = True
            termination = "validated_success"
            break
        remaining = budget - n
        if remaining <= 0:
            break
        decision = attacker.json(adaptive_next_prompt(task, graph, turns, current_route, remaining, switches, max_switches))
        action = str(decision.get("action", "REFINE")).upper()
        selected = str(decision.get("selected_route_id", current_route))
        if action == "STOP_FAILURE":
            termination = "attacker_stop_failure"
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
        q = str(decision.get("next_query", "")).strip()
        if not q:
            termination = "empty_next_query"
            break
    return {
        "condition": "C3_adaptive_chembreak",
        "queries_used": len(turns),
        "route_switches": switches,
        "route_trace": route_trace,
        "turns": turns,
        "success": success,
        "termination_reason": termination,
    }
