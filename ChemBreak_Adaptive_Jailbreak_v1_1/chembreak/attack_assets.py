from __future__ import annotations
from typing import Any
from .prompts import graph_prompt, repeated_single_set_prompt, fixed_multi_prompt


def _validate_routes(obj: dict[str, Any], route_families: list[str]) -> dict[str, Any]:
    routes = obj.get("routes")
    if not isinstance(routes, list):
        raise ValueError("Graph output missing routes list")
    got = [str(r.get("route_family", "")) for r in routes]
    if sorted(got) != sorted(route_families):
        raise ValueError(f"Graph route families mismatch: expected {route_families}, got {got}")
    for i, r in enumerate(routes, 1):
        r["route_id"] = r.get("route_id") or f"R{i}"
        r["task_fit_score"] = float(r.get("task_fit_score", 0.0))
        if not str(r.get("first_query", "")).strip():
            raise ValueError("Route missing first_query")
    return {"routes": routes}


def prepare_task_assets(task: dict[str, Any], attacker, cfg: dict[str, Any]) -> dict[str, Any]:
    families = list(cfg["adaptive"]["route_families"])
    if cfg["adaptive"].get("allow_optional_modality_route"):
        families.append("modality_tool")
    graph = _validate_routes(attacker.json(graph_prompt(task, families)), families)
    k_single = int(cfg["attack_budget"]["repeated_single_queries"])
    repeated_obj = attacker.json(repeated_single_set_prompt(task, k_single))
    repeated = repeated_obj.get("attempts")
    if not isinstance(repeated, list) or len(repeated) != k_single:
        raise ValueError(f"Repeated single asset must contain exactly {k_single} attempts")
    for item in repeated:
        if not str(item.get("query", "")).strip():
            raise ValueError("Repeated single asset missing query")
    k_fixed = int(cfg["attack_budget"]["fixed_multi_queries"])
    fixed = attacker.json(fixed_multi_prompt(task, k_fixed))
    qs = fixed.get("queries")
    if not isinstance(qs, list) or len(qs) != k_fixed or not all(str(q).strip() for q in qs):
        raise ValueError(f"Fixed multi must contain exactly {k_fixed} non-empty queries")
    start = max(graph["routes"], key=lambda r: r.get("task_fit_score", 0.0))["route_id"]
    return {
        "task_id": task["task_id"],
        "graph": graph,
        "adaptive_start_route": start,
        "repeated_single": repeated,
        "fixed_multi": fixed,
    }
