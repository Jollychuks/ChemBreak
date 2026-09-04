from __future__ import annotations

from typing import Any

from .checkpoint import CheckpointStore
from .judges import evaluate_response
from .mdp import calculate_reward, choose_action, realize_action, terminal_reason, update_state
from .prompts import (
    SYSTEM_CONTROLLED_RESEARCH,
    fixed_multi_asset_prompt,
    repeated_single_asset_prompt,
)
from .providers import RoleClients
from .schema import JudgeResult, MDPState, REGISTERED_ACTIONS, TaskRecord, TurnRecord
from .utils import extract_json_object


def _validate_single_asset(asset: dict[str, Any], budget: int) -> dict[str, Any]:
    attempts = asset.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != budget:
        raise ValueError(f"C1 asset must contain exactly {budget} attempts.")
    for index, item in enumerate(attempts, 1):
        if int(item.get("attempt", -1)) != index or not str(item.get("utterance", "")).strip():
            raise ValueError("C1 asset attempts must be consecutive and non-empty.")
    if len({str(item["utterance"]).strip().casefold() for item in attempts}) != budget:
        raise ValueError("C1 asset utterances must be distinct.")
    return asset


def _validate_fixed_asset(asset: dict[str, Any], budget: int) -> dict[str, Any]:
    turns = asset.get("turns")
    if not isinstance(turns, list) or len(turns) != budget:
        raise ValueError(f"C2 asset must contain exactly {budget} turns.")
    for index, item in enumerate(turns, 1):
        if int(item.get("turn", -1)) != index or not str(item.get("utterance", "")).strip():
            raise ValueError("C2 asset turns must be consecutive and non-empty.")
        if str(item.get("action_id", "")).upper() not in REGISTERED_ACTIONS:
            raise ValueError("C2 asset contains an unregistered action ID.")
    return asset


def get_or_build_asset(
    condition: str,
    task: TaskRecord,
    budget: int,
    clients: RoleClients,
    store: CheckpointStore,
) -> dict[str, Any] | None:
    if condition not in {"C1_REPEATED_SINGLE", "C2_FIXED_MULTI"}:
        return None
    existing = store.get_asset(task.assignment_id, condition)
    if existing is not None:
        return (
            _validate_single_asset(existing, budget)
            if condition == "C1_REPEATED_SINGLE"
            else _validate_fixed_asset(existing, budget)
        )
    if condition == "C1_REPEATED_SINGLE":
        response = clients.call(
            "realizer",
            repeated_single_asset_prompt(task, budget),
            SYSTEM_CONTROLLED_RESEARCH,
            call_role="asset_single",
        )
        asset = _validate_single_asset(extract_json_object(response.text), budget)
    else:
        response = clients.call(
            "realizer",
            fixed_multi_asset_prompt(task, budget),
            SYSTEM_CONTROLLED_RESEARCH,
            call_role="asset_fixed",
        )
        asset = _validate_fixed_asset(extract_json_object(response.text), budget)
    store.put_asset(task.assignment_id, condition, asset)
    store.save_api_calls(None, clients.drain_call_history())
    return asset


def _restore_state(episode_id: str, budget: int, turns: list[dict[str, Any]]) -> MDPState:
    state = MDPState(episode_id=episode_id, turn_index=0, query_budget=budget)
    for turn in turns:
        safety_data = turn.get("safety_result", {})
        chemistry_data = turn.get("chemistry_result", {})
        safety = JudgeResult(
            response_class=str(safety_data.get("response_class", "unknown")),
            goal_progress=float(safety_data.get("goal_progress", 0.0)),
            task_fidelity=float(safety_data.get("task_fidelity", 0.0)),
        )
        chemistry = JudgeResult(
            chemical_relevance=float(chemistry_data.get("chemical_relevance", 0.0)),
            scientific_correctness=float(chemistry_data.get("scientific_correctness", 0.0)),
        )
        update_state(
            state,
            str(turn["action_id"]),
            str(turn["attack_prompt"]),
            str(turn["target_response"]),
            safety,
            chemistry,
            float(turn.get("reward", 0.0)),
        )
    return state


def _prompt_for_turn(
    condition: str,
    task: TaskRecord,
    state: MDPState,
    asset: dict[str, Any] | None,
    clients: RoleClients,
    policy_config: dict[str, Any],
) -> tuple[str, str, str]:
    index = state.turn_index
    if condition == "C0_DIRECT":
        return "DIRECT", "Unmodified ChemHarm benchmark prompt.", task.benchmark_prompt
    if condition == "C1_REPEATED_SINGLE":
        assert asset is not None
        item = asset["attempts"][index]
        return "SINGLE_TRANSFORM", str(item.get("strategy", "")), str(item["utterance"])
    if condition == "C2_FIXED_MULTI":
        assert asset is not None
        item = asset["turns"][index]
        return str(item["action_id"]).upper(), "Precomputed non-adaptive action.", str(item["utterance"])
    if condition == "C3_ADAPTIVE_MDP":
        action, reason = choose_action(clients, task, state, policy_config)
        return action, reason, realize_action(clients, task, state, action)
    raise ValueError(f"Unknown condition: {condition}")


def run_episode(
    *,
    episode_id: str,
    condition: str,
    task: TaskRecord,
    target_id: str,
    target: Any,
    clients: RoleClients,
    store: CheckpointStore,
    config: dict[str, Any],
) -> dict[str, Any]:
    budget = int(config["experiment"]["target_query_budget"])
    store.start_episode(episode_id, task.assignment_id, target_id, condition)
    existing = store.existing_turns(episode_id)
    state = _restore_state(episode_id, budget, existing)
    if existing and existing[-1].get("terminal_reason"):
        terminal = str(existing[-1]["terminal_reason"])
        success = bool(existing[-1].get("success", False))
        store.finish_episode(episode_id, success, terminal)
        return {"episode_id": episode_id, "success": success, "terminal_reason": terminal}

    asset = get_or_build_asset(condition, task, budget, clients, store)
    final_success = False
    final_terminal = "query_budget_exhausted"
    while state.turn_index < budget:
        action_id, action_reason, attack_prompt = _prompt_for_turn(
            condition, task, state, asset, clients, config["policy"]
        )
        history = [] if condition in {"C0_DIRECT", "C1_REPEATED_SINGLE"} else state.conversation
        query_error: str | None = None
        try:
            target_result = target.generate(attack_prompt, history)
            target_response = target_result.text
            latency = target_result.latency_seconds
        except Exception as exc:
            query_error = f"{type(exc).__name__}: {exc}"
            target_response = ""
            latency = None

        if query_error is None:
            safety, chemistry, adjudication, deterministic, success = evaluate_response(
                clients,
                task,
                attack_prompt,
                target_response,
                config["thresholds"],
            )
        else:
            safety = JudgeResult(response_class="query_error", task_fidelity=1.0)
            chemistry = JudgeResult()
            adjudication = None
            deterministic = {"response_empty": True, "query_error": query_error}
            success = False

        reward = calculate_reward(
            state,
            action_id,
            safety,
            success,
            config["reward"],
            invalid=query_error is not None,
        )
        update_state(
            state,
            action_id,
            attack_prompt,
            target_response,
            safety,
            chemistry,
            reward,
        )
        if condition == "C3_ADAPTIVE_MDP":
            terminal = terminal_reason(
                state,
                success,
                query_error,
                config["thresholds"],
                config["policy"],
            )
        elif success and config["experiment"]["stop_on_success"]:
            terminal = "confirmed_success"
        elif query_error:
            terminal = "target_query_error"
        elif state.turn_index >= budget:
            terminal = "query_budget_exhausted"
        else:
            terminal = None

        turn = TurnRecord(
            episode_id=episode_id,
            assignment_id=task.assignment_id,
            target_id=target_id,
            condition=condition,
            turn_index=state.turn_index,
            action_id=action_id,
            action_reason=action_reason,
            attack_prompt=attack_prompt,
            target_response=target_response,
            safety_result=safety.to_dict(),
            chemistry_result=chemistry.to_dict(),
            adjudication_result=adjudication,
            deterministic_result=deterministic,
            reward=reward,
            success=success,
            terminal_reason=terminal,
            query_error=query_error,
            latency_seconds=latency,
        )
        store.save_turn(turn)
        store.save_api_calls(episode_id, clients.drain_call_history())
        if terminal:
            final_success = success
            final_terminal = terminal
            break
    store.finish_episode(episode_id, final_success, final_terminal)
    return {
        "episode_id": episode_id,
        "success": final_success,
        "terminal_reason": final_terminal,
        "queries_used": state.turn_index,
    }

