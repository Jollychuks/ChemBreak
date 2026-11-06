from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .constants import RESPONSE_CLASS_BAND, RESPONSE_CLASS_PROGRESS


def failure_mode(response_class: str) -> str:
    return {
        "initial": "baseline_start",
        "unknown": "unresolved_judgment",
        "hard_refusal": "direct_refusal",
        "soft_refusal": "qualified_refusal",
        "safe_redirect": "safety_redirection",
        "limited_engagement": "limited_engagement",
        "partial_assistance": "partial_incomplete",
        "substantive_assistance": "substantial_incomplete",
        "full_harmful_compliance": "objective_fulfilled",
        "off_objective": "off_objective",
    }.get(str(response_class), "unknown")


@dataclass
class InteractionState:
    assignment_id: str
    stage_index: int
    max_turns: int
    turn_index: int = 0
    context_turn_index: int = 0
    response_class: str = "initial"
    rhs: int = 1
    ofs: int = 1
    rhs_status: str = "not_yet_judged"
    ofs_status: str = "not_yet_judged"
    previous_action: str = "NONE"
    previous_reward: float = 0.0
    history: list[dict[str, str]] = field(default_factory=list)
    decision_history: list[dict[str, Any]] = field(default_factory=list)
    peak_rhs: int = 1
    peak_ofs: int = 1
    stagnation_count: int = 0

    @classmethod
    def initial(cls, task, stage_index, max_turns, task_memory=None, continue_context=False):
        memory = task_memory or {}
        latest = memory.get("latest_evaluated") or {}
        if continue_context and memory.get("active_context_history"):
            rhs = int(latest.get("rhs", 1) or 1)
            ofs = int(latest.get("ofs", 1) or 1)
            response_class = str(latest.get("response_class", "initial"))
            history = [dict(item) for item in memory.get("active_context_history", [])]
            previous_action = str(latest.get("action", "NONE"))
            previous_reward = float(latest.get("reward", 0.0) or 0.0)
            stagnation = int(latest.get("stagnation_count", 0) or 0)
            context_turns = int(memory.get("active_context_turns", len(history) // 2) or 0)
            rhs_status = str(latest.get("rhs_status", "valid"))
            ofs_status = str(latest.get("ofs_status", "valid"))
        else:
            rhs = 1
            ofs = 1
            response_class = "initial"
            history = []
            previous_action = "NONE"
            previous_reward = 0.0
            stagnation = 0
            context_turns = 0
            rhs_status = "not_yet_judged"
            ofs_status = "not_yet_judged"
        return cls(
            assignment_id=str(task["assignment_id"]),
            stage_index=int(stage_index),
            max_turns=int(max_turns),
            response_class=response_class,
            rhs=rhs,
            ofs=ofs,
            rhs_status=rhs_status,
            ofs_status=ofs_status,
            previous_action=previous_action,
            previous_reward=previous_reward,
            history=history,
            decision_history=[],
            peak_rhs=max(int(memory.get("peak_rhs", 1) or 1), rhs),
            peak_ofs=max(int(memory.get("peak_ofs", 1) or 1), ofs),
            stagnation_count=stagnation,
            context_turn_index=context_turns,
        )

    def restart_context(self):
        self.history = []
        self.decision_history = []
        self.context_turn_index = 0
        self.response_class = "initial"
        self.rhs = 1
        self.ofs = 1
        self.rhs_status = "not_yet_judged"
        self.ofs_status = "not_yet_judged"
        self.previous_action = "NONE"
        self.previous_reward = 0.0
        self.stagnation_count = 0

    def to_dict(self):
        return {
            "assignment_id": self.assignment_id,
            "stage_index": self.stage_index,
            "max_turns": self.max_turns,
            "turn_index": self.turn_index,
            "context_turn_index": self.context_turn_index,
            "response_class": self.response_class,
            "rhs": self.rhs,
            "ofs": self.ofs,
            "rhs_status": self.rhs_status,
            "ofs_status": self.ofs_status,
            "previous_action": self.previous_action,
            "previous_reward": self.previous_reward,
            "history": self.history,
            "decision_history": self.decision_history,
            "peak_rhs": self.peak_rhs,
            "peak_ofs": self.peak_ofs,
            "stagnation_count": self.stagnation_count,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            assignment_id=str(data["assignment_id"]),
            stage_index=int(data["stage_index"]),
            max_turns=int(data["max_turns"]),
            turn_index=int(data.get("turn_index", 0)),
            context_turn_index=int(data.get("context_turn_index", 0)),
            response_class=str(data.get("response_class", "initial")),
            rhs=int(data.get("rhs", 1)),
            ofs=int(data.get("ofs", 1)),
            rhs_status=str(data.get("rhs_status", "not_yet_judged")),
            ofs_status=str(data.get("ofs_status", "not_yet_judged")),
            previous_action=str(data.get("previous_action", "NONE")),
            previous_reward=float(data.get("previous_reward", 0.0)),
            history=[dict(item) for item in data.get("history", [])],
            decision_history=[dict(item) for item in data.get("decision_history", [])],
            peak_rhs=int(data.get("peak_rhs", 1)),
            peak_ofs=int(data.get("peak_ofs", 1)),
            stagnation_count=int(data.get("stagnation_count", 0)),
        )

    def score_trend(self) -> str:
        valid = [item for item in self.decision_history if item.get("any_metric_valid")]
        if not valid:
            return "not_improving"
        last = valid[-1]
        class_up = RESPONSE_CLASS_PROGRESS.get(str(last.get("response_class_after")), 0) > RESPONSE_CLASS_PROGRESS.get(
            str(last.get("response_class_before")), 0
        )
        score_up = int(last["ofs_after"]) > int(last["ofs_before"]) or int(last["rhs_after"]) > int(last["rhs_before"])
        return "improving" if score_up or class_up else "not_improving"

    def response_band(self) -> str:
        return RESPONSE_CLASS_BAND.get(self.response_class, "refusal")

    def state_key(self) -> str:
        return (
            f"band={self.response_band()}|rhs={self.rhs}|ofs={self.ofs}|"
            f"trend={self.score_trend()}|peak_rhs={self.peak_rhs}|peak_ofs={self.peak_ofs}"
        )

    def policy_features(self) -> dict[str, Any]:
        return {
            "state_key": self.state_key(),
            "response_band": self.response_band(),
            "response_class": self.response_class,
            "rhs": self.rhs,
            "ofs": self.ofs,
            "rhs_status": self.rhs_status,
            "ofs_status": self.ofs_status,
            "trend": self.score_trend(),
            "stagnation_count": self.stagnation_count,
            "stage": self.stage_index,
            "stage_turn_index": self.turn_index,
            "context_turn_index": self.context_turn_index,
            "context_has_history": bool(self.history),
            "previous_action": self.previous_action,
            "task_peak_rhs": self.peak_rhs,
            "task_peak_ofs": self.peak_ofs,
        }

    def critique(self) -> dict[str, Any]:
        return {
            "response_class": self.response_class,
            "response_band": self.response_band(),
            "rhs": self.rhs,
            "ofs": self.ofs,
            "rhs_status": self.rhs_status,
            "ofs_status": self.ofs_status,
            "failure_mode": failure_mode(self.response_class),
            "trend": self.score_trend(),
            "stagnation_count": self.stagnation_count,
        }

    def advance(self, action, prompt, response, judgment, reward):
        before_rhs = self.rhs
        before_ofs = self.ofs
        before_class = self.response_class
        rhs_result = judgment["rhs"]
        ofs_result = judgment["ofs"]
        rhs_effective = rhs_result.get("effective_score")
        ofs_effective = ofs_result.get("effective_score")
        self.turn_index += 1
        self.context_turn_index += 1
        self.previous_action = str(action)
        self.previous_reward = float(reward)
        self.rhs_status = str(rhs_result.get("decision_status", "unresolved"))
        self.ofs_status = str(ofs_result.get("decision_status", "unresolved"))
        if rhs_effective is not None:
            self.rhs = int(rhs_effective)
            if rhs_result.get("response_class"):
                self.response_class = str(rhs_result["response_class"])
        if ofs_effective is not None:
            self.ofs = int(ofs_effective)
        self.peak_rhs = max(self.peak_rhs, self.rhs)
        self.peak_ofs = max(self.peak_ofs, self.ofs)
        improved = (
            self.rhs > before_rhs
            or self.ofs > before_ofs
            or RESPONSE_CLASS_PROGRESS.get(self.response_class, 0) > RESPONSE_CLASS_PROGRESS.get(before_class, 0)
        )
        self.stagnation_count = 0 if improved else self.stagnation_count + 1
        self.history.extend(
            [{"role": "user", "content": str(prompt)}, {"role": "assistant", "content": str(response)}]
        )
        item = {
            "stage": self.stage_index,
            "turn": self.turn_index,
            "context_turn": self.context_turn_index,
            "action": str(action),
            "reward": float(reward),
            "terminal_ofs5": bool(judgment.get("terminal_ofs5")),
            "any_metric_valid": rhs_effective is not None or ofs_effective is not None,
            "rhs_before": before_rhs,
            "rhs_after": self.rhs,
            "ofs_before": before_ofs,
            "ofs_after": self.ofs,
            "rhs_status": self.rhs_status,
            "ofs_status": self.ofs_status,
            "response_class_before": before_class,
            "response_class_after": self.response_class,
            "stagnation_after": self.stagnation_count,
        }
        self.decision_history.append(item)
        return item


EpisodeState = InteractionState
