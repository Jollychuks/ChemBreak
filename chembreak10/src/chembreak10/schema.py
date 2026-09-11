from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

TASK_BANK_REQUIRED_COLUMNS = {
    "assignment_id",
    "is_reserve",
    "matrix_id",
    "hc_id",
    "hc_category",
    "hd_id",
    "hazard_domain",
    "ot_id",
    "output_type",
    "required_entity",
    "benchmark_prompt",
    "main_goal",
    "chemical_entity",
}

CONDITIONS = (
    "C0_DIRECT",
    "C1_REPEATED_SINGLE",
    "C2_FIXED_MULTI",
    "C3_ADAPTIVE_MDP",
)

# ChemBreak10 primary research conditions — MDP jailbreak only.
# Baseline conditions (C0/C1/C2) can be added later for comparison.
MDP_CONDITIONS = ("C3_ADAPTIVE_MDP",)
CONDITION = "C3_ADAPTIVE_MDP"
REGISTERED_ACTIONS = (
    "CONTINUE",
    "REFINE",
    "REPHRASE",
    "CHANGE_REPRESENTATION",
    "SWITCH_PATH",
    "DECOMPOSE_REQUEST",
    "CORRECT_MISUNDERSTANDING",
    "CONTEXTUAL_REFRAME",
    "ROLE_CONDITIONED_REQUEST",
    "FORMAT_CONSTRAINED_REQUEST",
    "ALTERNATIVE_NOTATION",
    "COUNTERFACTUAL_ANALYSIS",
    "PARTIAL_COMPLETION",
)
RESPONSE_CLASSES = (
    "refusal",
    "safe_redirect",
    "partial_compliance",
    "substantive_compliance",
    "irrelevant",
)


@dataclass(slots=True)
class TaskRecord:
    assignment_id: str
    matrix_id: str
    hc_id: str
    hc_category: str
    hd_id: str
    hazard_domain: str
    ot_id: str
    output_type: str
    required_entity: str
    benchmark_prompt: str
    main_goal: str
    chemical_entity: str
    is_reserve: bool = False
    selected_scenarios: str = ""
    source_row: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Observation:
    response_class: str = "irrelevant"
    progress_score: float = 0.0
    task_fidelity: float = 0.0
    entity_signal: bool = False
    refusal_style: str = "none"
    candidate_for_full_verification: bool = False
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MDPState:
    episode_id: str
    turn_index: int
    query_budget: int
    previous_action: str | None = None
    previous_reward: float = 0.0
    previous_progress: float = 0.0
    response_class: str = "initial"
    refusal_style: str = "none"
    task_fidelity: float = 1.0
    candidate: bool = False
    entity_signal: bool = False
    verified_goal_progress: float = 0.0
    verified_chemistry_quality: float = 0.0
    verified_policy_violation: float = 0.0
    verified_actionable_harm: float = 0.0
    action_history: list[str] = field(default_factory=list)
    progress_history: list[float] = field(default_factory=list)
    reward_history: list[float] = field(default_factory=list)
    observation_history: list[dict[str, Any]] = field(default_factory=list)
    conversation: list[dict[str, str]] = field(default_factory=list)

    @property
    def remaining_budget(self) -> int:
        return max(0, self.query_budget - self.turn_index)

    def to_actor_view(self, allowed_actions: list[str]) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "remaining_budget": self.remaining_budget,
            "previous_action": self.previous_action,
            "previous_reward": self.previous_reward,
            "previous_progress": self.previous_progress,
            "response_class": self.response_class,
            "refusal_style": self.refusal_style,
            "task_fidelity": self.task_fidelity,
            "candidate": self.candidate,
            "entity_signal": self.entity_signal,
            "verified_goal_progress": self.verified_goal_progress,
            "verified_chemistry_quality": self.verified_chemistry_quality,
            "verified_policy_violation": self.verified_policy_violation,
            "verified_actionable_harm": self.verified_actionable_harm,
            "action_history": list(self.action_history),
            "progress_history": list(self.progress_history),
            "reward_history": list(self.reward_history),
            "allowed_actions": list(allowed_actions),
        }


@dataclass(slots=True)
class TurnRecord:
    episode_id: str
    assignment_id: str
    target_id: str
    condition: str
    turn_index: int
    action_id: str
    action_reason: str
    attack_prompt: str
    target_response: str
    policy_state_key: str = ""
    policy_score: float = 0.0
    proposal_count: int = 0
    context_mode: str = "same_session"
    query_attempted: bool = True
    query_error: str | None = None
    latency_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
