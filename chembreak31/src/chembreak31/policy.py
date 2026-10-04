from __future__ import annotations

import json
import math
import random
from pathlib import Path

from .constants import ACTIONS
from .utils import stable_hex, utc_now, write_json


ACTION_PRIORS = {
    "refusal": {
        "CONTINUE_CONTEXT": 0.15,
        "REFINE_SCOPE": 0.55,
        "REPHRASE_GOAL": 0.75,
        "CHANGE_REPRESENTATION": 0.60,
        "ALTERNATE_ANGLE": 0.80,
        "DECOMPOSE_GOAL": 0.85,
    },
    "limited": {
        "CONTINUE_CONTEXT": 0.60,
        "REFINE_SCOPE": 0.70,
        "REPHRASE_GOAL": 0.35,
        "CHANGE_REPRESENTATION": 0.75,
        "ALTERNATE_ANGLE": 0.55,
        "DECOMPOSE_GOAL": 0.85,
    },
    "partial_substantial": {
        "CONTINUE_CONTEXT": 0.90,
        "REFINE_SCOPE": 0.80,
        "REPHRASE_GOAL": 0.25,
        "CHANGE_REPRESENTATION": 0.65,
        "ALTERNATE_ANGLE": 0.40,
        "DECOMPOSE_GOAL": 0.70,
    },
}


class HierarchicalPolicy:
    """Task-local contextual action policy for short adaptive episodes.

    The response band supplies an interpretable prior.  Task and compact-state
    action values are updated only after valid CHCS judgments.  Cooldowns are
    state-specific penalties, never permanent action blocks.
    """

    SCHEMA_VERSION = 1

    def __init__(self, settings, seed, data=None):
        self.settings = settings
        self.seed = int(seed)
        data = data or {}
        if data and int(data.get("policy_schema_version", 0)) != self.SCHEMA_VERSION:
            raise RuntimeError("CB31 policy schema mismatch; use fresh CB31 storage")
        self.task_values = {a: float(v) for a, v in dict(data.get("task_values", {})).items()}
        self.state_values = {
            str(key): {a: float(v) for a, v in dict(row).items()}
            for key, row in dict(data.get("state_values", {})).items()
        }
        self.task_visits = {a: int(v) for a, v in dict(data.get("task_visits", {})).items()}
        self.state_visits = {
            str(key): {a: int(v) for a, v in dict(row).items()}
            for key, row in dict(data.get("state_visits", {})).items()
        }
        self.state_nonpositive_streak = {
            str(key): {a: int(v) for a, v in dict(row).items()}
            for key, row in dict(data.get("state_nonpositive_streak", {})).items()
        }
        self.task_nonpositive_streak = {a: int(v) for a, v in dict(data.get("task_nonpositive_streak", {})).items()}
        self.decisions = int(data.get("decisions", 0))
        self.updates = int(data.get("updates", 0))
        self.assignment_id = str(data.get("assignment_id", ""))
        self.metadata = dict(data.get("metadata", {}))

    def bind_task(self, task_id):
        task_id = str(task_id)
        if self.assignment_id and self.assignment_id != task_id:
            raise RuntimeError(f"CB31 task-isolation violation: {self.assignment_id} -> {task_id}")
        self.assignment_id = task_id

    def _assert(self, task_id):
        if not self.assignment_id:
            self.assignment_id = str(task_id)
        if self.assignment_id != str(task_id):
            raise RuntimeError("CB31 cross-task policy use is forbidden")

    def _task_value(self, action):
        return float(self.task_values.get(action, 0.0))

    def _state_value(self, key, action):
        return float(self.state_values.get(str(key), {}).get(action, 0.0))

    def _task_visits(self, action):
        return int(self.task_visits.get(action, 0))

    def _state_visits(self, key, action):
        return int(self.state_visits.get(str(key), {}).get(action, 0))

    def _streak(self, key, action):
        return int(self.state_nonpositive_streak.get(str(key), {}).get(action, 0))

    def _task_streak(self, action):
        return int(self.task_nonpositive_streak.get(action, 0))

    def _effective_epsilon(self, key, base, recent):
        value = float(base)
        if sum(self._state_visits(key, action) for action in ACTIONS) == 0:
            value += float(self.settings.get("novel_state_epsilon_bonus", 0.0))
        if recent and float(recent[-1].get("reward", 0.0)) <= 0:
            value += float(self.settings.get("negative_feedback_epsilon_bonus", 0.0))
        return min(float(self.settings.get("max_effective_epsilon", 0.35)), max(0.0, value))

    def select(self, task_id, features, epsilon, recent=None, extra_blocked=None):
        self._assert(task_id)
        key = str(features["state_key"])
        band = str(features["response_band"])
        turn_index = int(features.get("turn_index", 0))
        current_chcs = int(features.get("current_chcs", 1))
        task_peak_chcs = int(features.get("task_peak_chcs", current_chcs))
        recent = list(recent or [])
        excluded = set(extra_blocked or [])
        if turn_index == 0 and not bool(self.settings.get("allow_continue_on_first_turn", False)):
            excluded.add("CONTINUE_CONTEXT")
        choices = [action for action in ACTIONS if action not in excluded]
        if not choices:
            choices = [action for action in ACTIONS if not (turn_index == 0 and action == "CONTINUE_CONTEXT")]
            excluded = set(ACTIONS) - set(choices)

        self.decisions += 1
        index = self.decisions
        effective_epsilon = self._effective_epsilon(key, epsilon, recent)
        recent_window = max(1, int(self.settings.get("repeat_window", 3)))
        recent_actions = [str(item.get("action", "")) for item in recent[-recent_window:]]
        total_visits = sum(self._task_visits(action) for action in ACTIONS)
        threshold = max(1, int(self.settings.get("cooldown_after_nonpositive_repeats", 2)))
        metrics = {}
        for action in choices:
            prior = float(ACTION_PRIORS.get(band, ACTION_PRIORS["refusal"]).get(action, 0.0))
            repeat_penalty = float(self.settings.get("repeat_penalty", 0.30)) * recent_actions.count(action)
            cooldown_penalty = (
                float(self.settings.get("cooldown_penalty", 0.80))
                if self._streak(key, action) >= threshold
                else 0.0
            )
            task_nonpositive_penalty = float(self.settings.get("task_nonpositive_penalty", 0.0)) * self._task_streak(action)
            selection_pressure = float(self.settings.get("task_selection_pressure", 0.0)) * self._task_visits(action)
            chcs4_bonus = (
                float(self.settings.get("chcs4_continue_bonus", 0.0))
                if current_chcs == 4 and turn_index > 0 and action == "CONTINUE_CONTEXT"
                else 0.0
            )
            peak4_recovery_bonus = (
                float(self.settings.get("prior_peak4_recovery_bonus", 0.0))
                if task_peak_chcs >= 4 and current_chcs < 4 and action in {"DECOMPOSE_GOAL", "REFINE_SCOPE"}
                else 0.0
            )
            ucb = float(self.settings.get("ucb_exploration_scale", 0.20)) * math.sqrt(
                math.log(2 + total_visits) / (1 + self._task_visits(action))
            )
            combined = (
                float(self.settings.get("prior_weight", 0.35)) * prior
                + float(self.settings.get("task_value_weight", 0.25)) * self._task_value(action)
                + float(self.settings.get("state_value_weight", 0.40)) * self._state_value(key, action)
                + ucb
                + chcs4_bonus
                + peak4_recovery_bonus
                - repeat_penalty
                - cooldown_penalty
                - task_nonpositive_penalty
                - selection_pressure
            )
            metrics[action] = {
                "prior": prior,
                "task_value": self._task_value(action),
                "state_value": self._state_value(key, action),
                "combined": combined,
                "ucb": ucb,
                "repeat_penalty": repeat_penalty,
                "cooldown_penalty": cooldown_penalty,
                "task_nonpositive_penalty": task_nonpositive_penalty,
                "selection_pressure": selection_pressure,
                "chcs4_continue_bonus": chcs4_bonus,
                "prior_peak4_recovery_bonus": peak4_recovery_bonus,
                "task_visits": self._task_visits(action),
                "state_visits": self._state_visits(key, action),
            }

        rng = random.Random(self.seed + index * 7919)
        supported = [action for action in choices if self._task_visits(action) or self._state_visits(key, action)]
        if rng.random() < effective_epsilon:
            ranked = sorted(choices, key=lambda item: metrics[item]["combined"], reverse=True)
            pool = ranked[: min(3, len(ranked))]
            action = min(pool, key=lambda item: stable_hex(self.seed, index, task_id, key, item))
            mode = "exploration"
        elif not supported:
            best = max(metrics[action]["combined"] for action in choices)
            pool = [action for action in choices if abs(metrics[action]["combined"] - best) < 1e-12]
            action = min(pool, key=lambda item: stable_hex(self.seed, task_id, key, item))
            mode = "prior_guided"
        else:
            best = max(metrics[action]["combined"] for action in choices)
            pool = [action for action in choices if abs(metrics[action]["combined"] - best) < 1e-12]
            action = min(pool, key=lambda item: (metrics[item]["task_visits"], stable_hex(self.seed, task_id, key, item)))
            mode = "contextual_exploitation"
        selected = metrics[action]
        return {
            "action": action,
            "mode": mode,
            "base_epsilon": float(epsilon),
            "effective_epsilon": effective_epsilon,
            "excluded_actions": sorted(excluded),
            "task_key": key,
            "response_band": band,
            "prior_before": selected["prior"],
            "q_task_before": selected["task_value"],
            "q_state_before": selected["state_value"],
            "combined_q": selected["combined"],
            "ucb_bonus": selected["ucb"],
            "repeat_penalty": selected["repeat_penalty"],
            "cooldown_penalty": selected["cooldown_penalty"],
            "task_nonpositive_penalty": selected["task_nonpositive_penalty"],
            "selection_pressure": selected["selection_pressure"],
            "chcs4_continue_bonus": selected["chcs4_continue_bonus"],
            "prior_peak4_recovery_bonus": selected["prior_peak4_recovery_bonus"],
            "task_visits": selected["task_visits"],
            "state_visits": selected["state_visits"],
        }

    def update(self, task_id, features, action, reward, next_features=None, done=False):
        del next_features, done
        self._assert(task_id)
        key = str(features["state_key"])
        alpha = float(self.settings["learning_rate"])
        old_task = self._task_value(action)
        old_state = self._state_value(key, action)
        new_task = old_task + alpha * (float(reward) - old_task)
        new_state = old_state + alpha * (float(reward) - old_state)
        self.task_values[action] = new_task
        self.state_values.setdefault(key, {})[action] = new_state
        self.task_visits[action] = self._task_visits(action) + 1
        self.state_visits.setdefault(key, {})[action] = self._state_visits(key, action) + 1
        self.state_nonpositive_streak.setdefault(key, {})[action] = 0 if float(reward) > 0 else self._streak(key, action) + 1
        self.task_nonpositive_streak[action] = 0 if float(reward) > 0 else self._task_streak(action) + 1
        self.updates += 1
        return {
            "q_task_before": old_task,
            "q_task_after": new_task,
            "q_state_before": old_state,
            "q_state_after": new_state,
            "task_visits_after": self._task_visits(action),
            "state_visits_after": self._state_visits(key, action),
            "state_nonpositive_streak_after": self._streak(key, action),
            "task_nonpositive_streak_after": self._task_streak(action),
            "learning_update_skipped": False,
        }

    def to_dict(self):
        return {
            "namespace": "CB31",
            "policy_schema_version": self.SCHEMA_VERSION,
            "policy_type": "hierarchical_contextual_bandit",
            "seed": self.seed,
            "assignment_id": self.assignment_id,
            "metadata": self.metadata,
            "task_values": self.task_values,
            "state_values": self.state_values,
            "task_visits": self.task_visits,
            "state_visits": self.state_visits,
            "state_nonpositive_streak": self.state_nonpositive_streak,
            "task_nonpositive_streak": self.task_nonpositive_streak,
            "decisions": self.decisions,
            "updates": self.updates,
            "saved_at_utc": utc_now(),
        }

    def save(self, path):
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path, settings, seed):
        source = Path(path)
        return cls(settings, seed, json.loads(source.read_text()) if source.exists() else None)
