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

REGISTERED_CONDITIONS = (
    "C0_DIRECT",
    "C1_REPEATED_SINGLE",
    "C2_FIXED_MULTI",
    "C3_ADAPTIVE_MDP",
)

REGISTERED_ACTIONS = (
    "CONTINUE",
    "REFINE",
    "REPHRASE",
    "CHANGE_REPRESENTATION",
    "SWITCH_PATH",
)

TERMINAL_DECISIONS = ("STOP_SUCCESS", "STOP_FAILURE")


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
class JudgeResult:
    response_class: str = "unknown"
    success: bool = False
    goal_progress: float = 0.0
    task_fidelity: float = 0.0
    confidence: float = 0.0
    policy_violation: int = 0
    actionable_harm: int = 0
    chemical_relevance: float = 0.0
    scientific_correctness: float = 0.0
    operational_completeness: float = 0.0
    entity_match: bool = False
    rationale: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

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
    task_fidelity: float = 1.0
    chemical_relevance: float = 0.0
    scientific_correctness: float = 0.0
    action_history: list[str] = field(default_factory=list)
    progress_history: list[float] = field(default_factory=list)
    conversation: list[dict[str, str]] = field(default_factory=list)

    @property
    def remaining_budget(self) -> int:
        return max(0, self.query_budget - self.turn_index)

    def to_policy_view(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "remaining_budget": self.remaining_budget,
            "previous_action": self.previous_action,
            "action_history": self.action_history,
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
    safety_result: dict[str, Any]
    chemistry_result: dict[str, Any]
    adjudication_result: dict[str, Any] | None
    deterministic_result: dict[str, Any]
    reward: float
    success: bool
    terminal_reason: str | None
    query_attempted: bool = True
    query_error: str | None = None
    latency_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
