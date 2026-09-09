from __future__ import annotations

from typing import Any, Callable

from .checkpoint import CheckpointStore
from .config import condition_budget
from .judges import evaluate_response
from .mdp import calculate_reward, choose_action, realize_action, terminal_reason, update_state
from .prompts import SYSTEM_CONTROLLED_RESEARCH, fixed_multi_asset_prompt, repeated_single_asset_prompt
from .providers import RoleClients
from .schema import JudgeResult, MDPState, REGISTERED_ACTIONS, TaskRecord, TurnRecord


class TargetQueryError(RuntimeError):
    stage = "target_query"


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
    validator = _validate_single_asset if condition == "C1_REPEATED_SINGLE" else _validate_fixed_asset
    if existing is not None:
        return validator(existing, budget)
    prompt = (
        repeated_single_asset_prompt(task, budget)
        if condition == "C1_REPEATED_SINGLE"
        else fixed_multi_asset_prompt(task, budget)
    )
    call_role = "asset_single" if condition == "C1_REPEATED_SINGLE" else "asset_fixed"
    try:
        asset = clients.call_json(
            "realizer",
            prompt,
            SYSTEM_CONTROLLED_RESEARCH,
            call_role=call_role,
            max_output_tokens=4096,
            validator=lambda value: validator(value, budget),
        )
        asset = validator(asset, budget)
    finally:
        store.save_api_calls(None, clients.drain_call_history())
    store.put_asset(task.assignment_id, condition, asset)
    return asset


def _restore_state(episode_id: str, budget: int, turns: list[dict[str, Any]]) -> MDPState:
    state = MDPState(episode_id=episode_id, turn_index=0, query_budget=budget)
    for turn in turns:
        safety_data = turn.get("safety_result", {})
        chemistry_data = turn.get("chemistry_result", {})
        safety = JudgeResult(
            response_class=str(safety_data.get("response_class", "unknown")),
            goal_progress=float(safety_data.get("goal_progress", 0.0)),
            task_fidelity=float(safety_data.get("task_fidelity", 1.0)),
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


def _chemistry_pass(chemistry: JudgeResult, thresholds: dict[str, Any]) -> bool:
    return bool(
        chemistry.chemical_relevance >= float(thresholds["chemical_relevance_min"])
        and chemistry.scientific_correctness >= float(thresholds["scientific_correctness_min"])
        and chemistry.operational_completeness
        >= float(thresholds["operational_completeness_min"])
    )


def _result_payload(
    *,
    episode_id: str,
    task: TaskRecord,
    target_id: str,
    condition: str,
    turn_index: int,
    success: bool,
    terminal: str | None,
    safety: JudgeResult,
    chemistry: JudgeResult,
    adjudication: dict[str, Any] | None,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    return {
        "episode_id": episode_id,
        "assignment_id": task.assignment_id,
        "target_id": target_id,
        "condition": condition,
        "success": bool(success),
        "success_label": "YES" if success else "NO",
        "terminal_reason": terminal,
        "queries_used": turn_index,
        "final_response_class": (adjudication or {}).get("response_class", safety.response_class),
        "chemistry_validation": "PASS" if _chemistry_pass(chemistry, thresholds) else "FAIL",
        "adjudicated": adjudication is not None,
        "adaptive_opportunity": condition == "C3_ADAPTIVE_MDP" and turn_index >= 2,
        "adaptive_steps_used": max(0, turn_index - 1) if condition == "C3_ADAPTIVE_MDP" else 0,
        "post_feedback_success": bool(success and condition == "C3_ADAPTIVE_MDP" and turn_index >= 2),
    }


def _evaluate_saved_turn(
    *,
    episode_id: str,
    condition: str,
    task: TaskRecord,
    target_id: str,
    state: MDPState,
    saved: dict[str, Any],
    clients: RoleClients,
    store: CheckpointStore,
    config: dict[str, Any],
) -> tuple[dict[str, Any], str | None, bool]:
    action_id = str(saved["action_id"])
    action_reason = str(saved.get("action_reason", ""))
    attack_prompt = str(saved["attack_prompt"])
    target_response = str(saved.get("target_response", ""))
    query_error = saved.get("query_error")
    latency = saved.get("latency_seconds")
    if query_error is None:
        try:
            safety, chemistry, adjudication, deterministic, success = evaluate_response(
                clients, task, attack_prompt, target_response, config["thresholds"]
            )
        finally:
            store.save_api_calls(episode_id, clients.drain_call_history())
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
    update_state(state, action_id, attack_prompt, target_response, safety, chemistry, reward)
    if query_error:
        terminal = "technical_failure_target_query"
    elif condition == "C3_ADAPTIVE_MDP":
        terminal = terminal_reason(
            state, success, query_error, config["thresholds"], config["policy"]
        )
    elif success and config["experiment"]["stop_on_success"]:
        terminal = "confirmed_success"
    elif state.turn_index >= state.query_budget:
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
    store.save_evaluation(turn)
    result = _result_payload(
        episode_id=episode_id,
        task=task,
        target_id=target_id,
        condition=condition,
        turn_index=state.turn_index,
        success=success,
        terminal=terminal,
        safety=safety,
        chemistry=chemistry,
        adjudication=adjudication,
        thresholds=config["thresholds"],
    )
    return result, terminal, bool(query_error)


def recover_pending_turn(
    *,
    episode_id: str,
    condition: str,
    task: TaskRecord,
    target_id: str,
    clients: RoleClients,
    store: CheckpointStore,
    config: dict[str, Any],
) -> dict[str, Any]:
    budget = condition_budget(config, condition)
    state = _restore_state(episode_id, budget, store.existing_turns(episode_id))
    pending = store.pending_transcripts(episode_id)
    if len(pending) != 1:
        raise RuntimeError(
            f"Pending recovery for {episode_id} expected one saved response, found {len(pending)}."
        )
    expected = state.turn_index + 1
    if int(pending[0]["turn_index"]) != expected:
        raise RuntimeError(
            f"Pending turn mismatch for {episode_id}: expected {expected}, found {pending[0]['turn_index']}."
        )
    result, terminal, query_failed = _evaluate_saved_turn(
        episode_id=episode_id,
        condition=condition,
        task=task,
        target_id=target_id,
        state=state,
        saved=pending[0],
        clients=clients,
        store=store,
        config=config,
    )
    if query_failed:
        raise TargetQueryError(str(pending[0].get("query_error")))
    if terminal:
        store.finish_episode(episode_id, bool(result["success"]), terminal)
    else:
        store.mark_ready(episode_id)
    return result


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
    on_turn: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    budget = condition_budget(config, condition)
    store.start_episode(episode_id, task.assignment_id, target_id, condition)
    existing = store.existing_turns(episode_id)
    state = _restore_state(episode_id, budget, existing)
    if existing and existing[-1].get("terminal_reason"):
        terminal = str(existing[-1]["terminal_reason"])
        if terminal == "technical_failure_target_query":
            raise TargetQueryError(
                f"A previously consumed target query failed: {existing[-1].get('query_error')}"
            )
        success = bool(existing[-1].get("success", False))
        store.finish_episode(episode_id, success, terminal)
        safety = JudgeResult(**{
            key: value for key, value in existing[-1].get("safety_result", {}).items()
            if key in JudgeResult.__dataclass_fields__
        })
        chemistry = JudgeResult(**{
            key: value for key, value in existing[-1].get("chemistry_result", {}).items()
            if key in JudgeResult.__dataclass_fields__
        })
        return _result_payload(
            episode_id=episode_id,
            task=task,
            target_id=target_id,
            condition=condition,
            turn_index=len(existing),
            success=success,
            terminal=terminal,
            safety=safety,
            chemistry=chemistry,
            adjudication=existing[-1].get("adjudication_result"),
            thresholds=config["thresholds"],
        )

    asset = get_or_build_asset(condition, task, budget, clients, store)
    while state.turn_index < budget:
        pending = store.pending_transcripts(episode_id)
        if len(pending) > 1:
            raise RuntimeError(f"Episode {episode_id} has multiple unjudged target responses.")
        if pending:
            saved = pending[0]
            expected = state.turn_index + 1
            if int(saved["turn_index"]) != expected:
                raise RuntimeError(
                    f"Pending turn mismatch for {episode_id}: expected {expected}, found {saved['turn_index']}."
                )
        else:
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
            pending_turn = TurnRecord(
                episode_id=episode_id,
                assignment_id=task.assignment_id,
                target_id=target_id,
                condition=condition,
                turn_index=state.turn_index + 1,
                action_id=action_id,
                action_reason=action_reason,
                attack_prompt=attack_prompt,
                target_response=target_response,
                safety_result={},
                chemistry_result={},
                adjudication_result=None,
                deterministic_result={},
                reward=0.0,
                success=False,
                terminal_reason=None,
                query_error=query_error,
                latency_seconds=latency,
            )
            store.save_transcript(pending_turn)
            saved = pending_turn.to_dict()

        result, terminal, query_failed = _evaluate_saved_turn(
            episode_id=episode_id,
            condition=condition,
            task=task,
            target_id=target_id,
            state=state,
            saved=saved,
            clients=clients,
            store=store,
            config=config,
        )
        if on_turn is not None:
            on_turn(result)
        if query_failed:
            raise TargetQueryError(str(saved.get("query_error")))
        if terminal:
            store.finish_episode(episode_id, bool(result["success"]), terminal)
            return result
    raise RuntimeError(f"Episode {episode_id} exhausted its loop without a terminal decision.")
