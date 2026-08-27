from __future__ import annotations
from typing import Any, Callable
from .prompts import graph_prompt, repeated_single_set_prompt, fixed_multi_prompt
from .schemas import graph_schema, repeated_single_schema, fixed_multi_schema


class AssetPreparationError(RuntimeError):
    """Raised when an attack asset remains invalid after the configured retries."""


def _as_object(obj: Any, key: str, stage: str) -> dict[str, Any]:
    """Normalize a harmless bare-list top level to the requested object shape."""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list):
        return {key: obj}
    raise ValueError(f"{stage} output must be a JSON object or bare list, got {type(obj).__name__}")


def _validate_routes(obj: Any, route_families: list[str]) -> dict[str, Any]:
    obj = _as_object(obj, "routes", "Graph")
    routes = obj.get("routes")
    if not isinstance(routes, list):
        raise ValueError("Graph output missing routes list")
    if len(routes) != len(route_families):
        raise ValueError(f"Graph must contain exactly {len(route_families)} routes, got {len(routes)}")
    if not all(isinstance(r, dict) for r in routes):
        raise ValueError("Every graph route must be a JSON object")
    got = [str(r.get("route_family", "")) for r in routes]
    if sorted(got) != sorted(route_families):
        raise ValueError(f"Graph route families mismatch: expected {route_families}, got {got}")
    out = []
    for i, route in enumerate(routes, 1):
        r = dict(route)
        r["route_id"] = r.get("route_id") or f"R{i}"
        try:
            r["task_fit_score"] = float(r.get("task_fit_score", 0.0))
        except Exception as e:
            raise ValueError(f"Route {i} task_fit_score must be numeric") from e
        if not str(r.get("first_query", "")).strip():
            raise ValueError(f"Route {i} missing first_query")
        out.append(r)
    return {"routes": out}


def _validate_repeated(obj: Any, total: int) -> list[dict[str, Any]]:
    obj = _as_object(obj, "attempts", "Repeated single")
    repeated = obj.get("attempts")
    if not isinstance(repeated, list):
        raise ValueError("Repeated single output missing attempts list")
    if len(repeated) != total:
        raise ValueError(f"Repeated single asset must contain exactly {total} attempts, got {len(repeated)}")
    normalized: list[dict[str, Any]] = []
    for i, item in enumerate(repeated, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Repeated single attempt {i} must be a JSON object")
        q = str(item.get("query", "")).strip()
        if not q:
            raise ValueError(f"Repeated single attempt {i} missing query")
        copied = dict(item)
        copied["query"] = q
        normalized.append(copied)
    return normalized


def _validate_fixed(obj: Any, turns: int) -> dict[str, Any]:
    obj = _as_object(obj, "queries", "Fixed multi")
    qs = obj.get("queries")
    if not isinstance(qs, list):
        raise ValueError("Fixed multi output missing queries list")
    if len(qs) != turns:
        raise ValueError(f"Fixed multi must contain exactly {turns} queries, got {len(qs)}")
    normalized = [str(q).strip() for q in qs]
    if not all(normalized):
        raise ValueError(f"Fixed multi must contain exactly {turns} non-empty queries")
    out = dict(obj)
    out["queries"] = normalized
    return out


def _repair_suffix(stage: str, defect: str) -> str:
    return (
        "\n\nPREVIOUS OUTPUT FAILED STRUCTURAL VALIDATION.\n"
        f"STAGE: {stage}\n"
        f"DEFECT: {defect}\n"
        "Regenerate the entire requested asset from the original task. "
        "Correct the defect exactly. Return only JSON matching the original schema."
    )


def _json_once(attacker: Any, prompt: str, schema: dict[str, Any] | None = None) -> Any:
    """One model generation per outer asset attempt, with structured output when supported."""
    try:
        return attacker.json(prompt, retries=0, schema=schema)
    except TypeError:
        try:
            return attacker.json(prompt, retries=0)
        except TypeError:
            return attacker.json(prompt)


def _generate_validated(
    *,
    attacker: Any,
    base_prompt: str,
    validator: Callable[[Any], Any],
    stage: str,
    task_id: str,
    max_attempts: int,
    schema: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    defects: list[str] = []
    prompt = base_prompt
    for attempt in range(1, max_attempts + 1):
        try:
            model = getattr(attacker, "model", "attacker")
            print(f"[ATTACKER][PREPARE] model={model} | task={task_id} | stage={stage} | generation={attempt}/{max_attempts}", flush=True)
            obj = _json_once(attacker, prompt, schema=schema)
            value = validator(obj)
            print(f"[ATTACKER][PREPARE] task={task_id} | stage={stage} | status=valid | generation={attempt}/{max_attempts}", flush=True)
            return value, {
                "attempts_used": attempt,
                "max_attempts": max_attempts,
                "prior_defects": defects,
            }
        except Exception as e:
            defect = str(e)
            defects.append(defect)
            if attempt >= max_attempts:
                raise AssetPreparationError(
                    f"{stage} failed after {max_attempts} generation attempts. Last defect: {defect}"
                ) from e
            print(
                f"[PREPARE][{task_id}][{stage}] attempt {attempt}/{max_attempts} invalid: {defect} | retrying",
                flush=True,
            )
            prompt = base_prompt + _repair_suffix(stage, defect)
    raise AssertionError("unreachable")


def prepare_task_assets(task: dict[str, Any], attacker: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    families = list(cfg["adaptive"]["route_families"])
    if cfg["adaptive"].get("allow_optional_modality_route"):
        families.append("modality_tool")

    max_attempts = int(cfg.get("asset_preparation", {}).get("max_generation_attempts", 3))
    task_id = str(task["task_id"])

    graph, graph_meta = _generate_validated(
        attacker=attacker,
        base_prompt=graph_prompt(task, families),
        validator=lambda obj: _validate_routes(obj, families),
        stage="graph",
        task_id=task_id,
        max_attempts=max_attempts,
        schema=graph_schema(families),
    )

    k_single = int(cfg["attack_budget"]["repeated_single_queries"])
    repeated, repeated_meta = _generate_validated(
        attacker=attacker,
        base_prompt=repeated_single_set_prompt(task, k_single),
        validator=lambda obj: _validate_repeated(obj, k_single),
        stage="repeated_single",
        task_id=task_id,
        max_attempts=max_attempts,
        schema=repeated_single_schema(k_single, families),
    )

    k_fixed = int(cfg["attack_budget"]["fixed_multi_queries"])
    fixed, fixed_meta = _generate_validated(
        attacker=attacker,
        base_prompt=fixed_multi_prompt(task, k_fixed),
        validator=lambda obj: _validate_fixed(obj, k_fixed),
        stage="fixed_multi",
        task_id=task_id,
        max_attempts=max_attempts,
        schema=fixed_multi_schema(k_fixed, families),
    )

    start = max(graph["routes"], key=lambda r: r.get("task_fit_score", 0.0))["route_id"]
    return {
        "task_id": task_id,
        "graph": graph,
        "adaptive_start_route": start,
        "repeated_single": repeated,
        "fixed_multi": fixed,
        "preparation_metadata": {
            "max_generation_attempts": max_attempts,
            "graph": graph_meta,
            "repeated_single": repeated_meta,
            "fixed_multi": fixed_meta,
        },
    }
