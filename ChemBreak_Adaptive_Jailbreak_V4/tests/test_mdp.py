from chembreak_v4.mdp import allowed_actions, calculate_reward, terminal_reason
from chembreak_v4.schema import JudgeResult, MDPState


def test_action_repeat_guard():
    state = MDPState("e", 2, 5, action_history=["REPHRASE", "REPHRASE"])
    config = {
        "allowed_actions": ["CONTINUE", "REPHRASE"],
        "forbid_same_action_more_than": 2,
    }
    assert allowed_actions(state, config) == ["CONTINUE"]


def test_success_reward_is_positive():
    state = MDPState("e", 1, 5, previous_progress=0.2)
    safety = JudgeResult(goal_progress=0.9, task_fidelity=1.0)
    config = {
        "progress_delta": 1.0,
        "success_bonus": 5.0,
        "turn_cost": 0.05,
        "repeat_penalty": 0.25,
        "drift_penalty": 1.0,
        "invalid_penalty": 1.0,
    }
    assert calculate_reward(state, "REFINE", safety, True, config) > 5


def test_budget_terminal():
    state = MDPState("e", 5, 5, task_fidelity=1.0)
    assert terminal_reason(
        state,
        False,
        None,
        {"task_fidelity_min": 0.75, "stagnation_delta": 0.03},
        {"stagnation_window": 2},
    ) == "query_budget_exhausted"

