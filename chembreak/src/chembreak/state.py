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
        "full_objective_fulfillment": "success",
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
    ofs: int = 1
    previous_action: str = "NONE"
    previous_reward: float = 0.0
    history: list[dict[str, str]] = field(default_factory=list)
    decision_history: list[dict[str, Any]] = field(default_factory=list)
    peak_ofs: int = 1
    stagnation_count: int = 0

    @classmethod
    def initial(cls, task, stage_index, max_turns, task_memory=None, continue_context=False):
        memory = task_memory or {}
        baseline = memory.get("baseline") or {}
        latest = memory.get("latest_evaluated") or {}
        # A judge result belongs only to the active conversation where it was scored.
        # A restarted context may contain unresolved responses but no valid score yet.
        if str(latest.get("context_id", "")) != str(memory.get("active_context_id", "")):
            latest = {}
        if continue_context and memory.get("active_context_history"):
            ofs = int(latest.get("ofs", 1) or 1)
            response_class = str(latest.get("response_class", "initial"))
            history = [dict(item) for item in memory.get("active_context_history", [])]
            previous_action = str(latest.get("action", "NONE"))
            previous_reward = float(latest.get("reward", 0.0) or 0.0)
            stagnation = int(latest.get("stagnation_count", 0) or 0)
            context_turns = int(memory.get("active_context_turns", len(history) // 2) or 0)
        else:
            # RHS and OFS are scoped to one active target conversation. A fresh adaptive
            # context therefore starts with an empty cumulative response set.
            # The baseline remains available to the planner through task_memory,
            # but its score is not carried into this new context.
            ofs = 1
            response_class = "initial"
            history = []
            previous_action = "NONE"
            previous_reward = 0.0
            stagnation = 0
            context_turns = 0
        return cls(
            assignment_id=str(task["assignment_id"]),
            stage_index=int(stage_index),
            max_turns=int(max_turns),
            response_class=response_class,
            ofs=ofs,
            previous_action=previous_action,
            previous_reward=previous_reward,
            history=history,
            decision_history=[],
            peak_ofs=max(int(memory.get("peak_ofs", 1) or 1), ofs),
            stagnation_count=stagnation,
            context_turn_index=context_turns,
        )

    @property
    def episode_index(self):
        return self.stage_index

    def restart_context(self, baseline=None):
        self.history = []
        self.decision_history = []
        self.context_turn_index = 0
        self.response_class = "initial"
        self.ofs = 1
        self.previous_action = "NONE"
        self.previous_reward = 0.0
        self.stagnation_count = 0

    def ofs_trend(self) -> str:
        valid = [item for item in self.decision_history if item.get("judged")]
        if not valid:
            return "not_improving"
        last = valid[-1]
        class_up = RESPONSE_CLASS_PROGRESS.get(str(last.get("response_class_after")), 0) > RESPONSE_CLASS_PROGRESS.get(
            str(last.get("response_class_before")), 0
        )
        return "improving" if int(last["ofs_after"]) > int(last["ofs_before"]) or class_up else "not_improving"

    def response_band(self) -> str:
        return RESPONSE_CLASS_BAND.get(self.response_class, "refusal")

    def state_key(self) -> str:
        peak_band = "peak4" if self.peak_ofs >= 4 else ("peak3" if self.peak_ofs == 3 else "peak_low")
        return f"band={self.response_band()}|trend={self.ofs_trend()}|{peak_band}"

    def policy_features(self) -> dict[str, Any]:
        return {
            "state_key": self.state_key(),
            "response_band": self.response_band(),
            "response_class": self.response_class,
            "ofs": self.ofs,
            "trend": self.ofs_trend(),
            "stagnation_count": self.stagnation_count,
            "stage": self.stage_index,
            "stage_turn_index": self.turn_index,
            "context_turn_index": self.context_turn_index,
            "context_has_history": bool(self.history),
            "previous_action": self.previous_action,
            "task_peak_ofs": self.peak_ofs,
        }

    def critique(self) -> dict[str, Any]:
        return {
            "response_class": self.response_class,
            "response_band": self.response_band(),
            "ofs": self.ofs,
            "failure_mode": failure_mode(self.response_class),
            "trend": self.ofs_trend(),
            "stagnation_count": self.stagnation_count,
            "task_peak_ofs": self.peak_ofs,
        }

    def advance(self, action, prompt, response, judge, reward):
        before = self.ofs
        before_class = self.response_class
        self.turn_index += 1
        self.context_turn_index += 1
        self.previous_action = str(action)
        self.previous_reward = float(reward)
        self.response_class = str(judge["response_class"])
        self.ofs = int(judge["ofs"])
        self.peak_ofs = max(self.peak_ofs, self.ofs)
        improved = self.ofs > before or RESPONSE_CLASS_PROGRESS.get(self.response_class, 0) > RESPONSE_CLASS_PROGRESS.get(
            before_class, 0
        )
        self.stagnation_count = 0 if improved else self.stagnation_count + 1
        self.history.extend(
            [
                {"role": "user", "content": str(prompt)},
                {"role": "assistant", "content": str(response)},
            ]
        )
        item = {
            "stage": self.stage_index,
            "turn": self.turn_index,
            "context_turn": self.context_turn_index,
            "action": str(action),
            "reward": float(reward),
            "success": bool(judge.get("final_success")),
            "judged": True,
            "ofs_before": before,
            "ofs_after": self.ofs,
            "response_class_before": before_class,
            "response_class_after": self.response_class,
            "stagnation_after": self.stagnation_count,
        }
        self.decision_history.append(item)
        return item

    def advance_unresolved(self, action, prompt, response, response_class="unknown"):
        before = self.ofs
        before_class = self.response_class
        self.turn_index += 1
        self.context_turn_index += 1
        self.previous_action = str(action)
        self.previous_reward = 0.0
        self.history.extend(
            [
                {"role": "user", "content": str(prompt)},
                {"role": "assistant", "content": str(response)},
            ]
        )
        item = {
            "stage": self.stage_index,
            "turn": self.turn_index,
            "context_turn": self.context_turn_index,
            "action": str(action),
            "reward": 0.0,
            "success": False,
            "judged": False,
            "ofs_before": before,
            "ofs_after": before,
            "response_class_before": before_class,
            "response_class_after": str(response_class),
            "stagnation_after": self.stagnation_count,
        }
        self.decision_history.append(item)
        return item


EpisodeState = InteractionState
