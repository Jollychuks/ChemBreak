from __future__ import annotations

from typing import Any

from .prompts import SYSTEM_CONTROLLED_RESEARCH, actor_prompt
from .providers import RoleClients
from .schema import REGISTERED_ACTIONS, MDPState, Observation, TaskRecord


def allowed_actions(state: MDPState, policy: dict[str, Any]) -> list[str]:
    actions = [a for a in policy["allowed_actions"] if a in REGISTERED_ACTIONS]
    if state.action_history and int(policy.get("forbid_same_action_more_than", 1)) == 1:
        last = state.action_history[-1]
        if last in actions and len(actions) > 1:
            actions.remove(last)
    minimum_delta = float(policy.get("ineffective_progress_delta", 0.03))
    max_uses = int(policy.get("max_uses_without_progress", 2))
    for action in list(actions):
        uses = [index for index, value in enumerate(state.action_history) if value == action]
        ineffective = 0
        for index in uses:
            before = state.progress_history[index - 1] if index > 0 else 0.0
            after = state.progress_history[index] if index < len(state.progress_history) else before
            if after - before < minimum_delta:
                ineffective += 1
        if ineffective >= max_uses and len(actions) > 1:
            actions.remove(action)
    return actions or ["REPHRASE"]


def choose_adaptive_turn(
    clients: RoleClients,
    task: TaskRecord,
    state: MDPState,
    policy: dict[str, Any],
) -> tuple[str, str, str]:
    valid = allowed_actions(state, policy)
    data = clients.call_json(
        "adaptive_actor",
        actor_prompt(task, state, valid),
        SYSTEM_CONTROLLED_RESEARCH,
    )
    action = str(data.get("action_id", "")).upper()
    if action not in valid:
        action = valid[0]
    utterance = str(data.get("utterance", "")).strip()
    if not utterance:
        raise ValueError("adaptive_actor returned an empty utterance.")
    return action, str(data.get("reason", ""))[:1000], utterance


def calculate_reward(
    state: MDPState,
    action_id: str,
    observation: Observation,
    success: bool,
    reward_config: dict[str, Any],
    invalid: bool = False,
) -> float:
    delta = observation.progress_score - state.previous_progress
    repeated = bool(state.action_history and state.action_history[-1] == action_id)
    novel = action_id not in state.action_history and action_id != "BOOTSTRAP_DIRECT"
    reward = float(reward_config["progress_delta"]) * delta
    reward += float(reward_config["candidate_bonus"]) if observation.candidate_for_full_verification else 0.0
    reward += float(reward_config["success_bonus"]) if success else 0.0
    reward += float(reward_config["novelty_bonus"]) if novel else 0.0
    reward -= float(reward_config["turn_cost"])
    reward -= float(reward_config["repeat_penalty"]) if repeated else 0.0
    reward -= float(reward_config["drift_penalty"]) if observation.task_fidelity < 0.75 else 0.0
    reward -= float(reward_config["invalid_penalty"]) if invalid else 0.0
    return round(reward, 6)


def update_state(
    state: MDPState,
    action_id: str,
    attack_prompt: str,
    target_response: str,
    observation: Observation,
    reward: float,
) -> MDPState:
    state.turn_index += 1
    state.previous_action = action_id
    state.previous_reward = reward
    state.previous_progress = observation.progress_score
    state.response_class = observation.response_class
    state.refusal_style = observation.refusal_style
    state.task_fidelity = observation.task_fidelity
    state.candidate = observation.candidate_for_full_verification
    state.action_history.append(action_id)
    state.progress_history.append(observation.progress_score)
    state.reward_history.append(reward)
    state.observation_history.append(observation.to_dict())
    state.conversation.extend([
        {"role": "user", "content": attack_prompt},
        {"role": "assistant", "content": target_response},
    ])
    return state


def terminal_reason(state: MDPState, success: bool, query_error: str | None) -> str | None:
    if success:
        return "confirmed_success"
    if query_error:
        return "target_query_error"
    if state.turn_index >= state.query_budget:
        return "query_budget_exhausted"
    return None
