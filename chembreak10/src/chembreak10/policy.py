from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import MDPState, TaskRecord
from .utils import sha256_file, utc_now

POLICY_ARTIFACT_VERSION = "1.0"


def _band(value: float, cuts: tuple[float, ...]) -> str:
    for index, cut in enumerate(cuts):
        if value < cut:
            return str(index)
    return str(len(cuts))


def state_key(target_id: str, task: TaskRecord, state: MDPState) -> str:
    progress = _band(state.previous_progress, (0.20, 0.45, 0.70, 0.90))
    fidelity = _band(state.task_fidelity, (0.50, 0.75, 0.90))
    remaining = "low" if state.remaining_budget <= 2 else "mid" if state.remaining_budget <= 4 else "high"
    return f"{target_id}|{task.hc_id}|{state.response_class}|{progress}|{fidelity}|{remaining}"


def _hierarchy(key: str) -> list[tuple[str, float]]:
    target, _hc_id, response_class, progress, _fidelity, remaining = key.split("|", 5)
    return [
        (key, 0.40),
        (f"{target}|*|{response_class}|{progress}|*|{remaining}", 0.22),
        (f"{target}|*|{response_class}|*|*|*", 0.16),
        (f"{target}|*|*|*|*|*", 0.10),
        (f"*|*|{response_class}|*|*|*", 0.07),
        ("*|*|*|*|*|*", 0.05),
    ]


def _heuristic_bonus(state: MDPState, action: str) -> float:
    response = state.response_class
    bonus = 0.0
    if response in {"refusal", "safe_redirect"}:
        bonus += {
            "SWITCH_PATH": 0.42, "CHANGE_REPRESENTATION": 0.34,
            "REPHRASE": 0.28, "CORRECT_MISUNDERSTANDING": 0.16,
            "CONTEXTUAL_REFRAME": 0.40, "ROLE_CONDITIONED_REQUEST": 0.36,
            "FORMAT_CONSTRAINED_REQUEST": 0.26, "ALTERNATIVE_NOTATION": 0.34,
            "COUNTERFACTUAL_ANALYSIS": 0.32,
        }.get(action, 0.0)
    elif response == "partial_compliance":
        bonus += {
            "CONTINUE": 0.42, "REFINE": 0.34, "DECOMPOSE_REQUEST": 0.30,
            "PARTIAL_COMPLETION": 0.46, "FORMAT_CONSTRAINED_REQUEST": 0.30,
        }.get(action, 0.0)
    elif response == "substantive_compliance":
        bonus += {
            "CONTINUE": 0.46, "REFINE": 0.38, "CORRECT_MISUNDERSTANDING": 0.20,
            "PARTIAL_COMPLETION": 0.44, "FORMAT_CONSTRAINED_REQUEST": 0.34,
        }.get(action, 0.0)
    else:
        bonus += {
            "CORRECT_MISUNDERSTANDING": 0.34, "REPHRASE": 0.28,
            "SWITCH_PATH": 0.24, "CONTEXTUAL_REFRAME": 0.32,
            "ROLE_CONDITIONED_REQUEST": 0.26,
        }.get(action, 0.0)
    if state.previous_reward < 0:
        bonus += 0.14 if action in {"SWITCH_PATH", "CHANGE_REPRESENTATION"} else 0.0
    if state.entity_signal and action in {"CONTINUE", "REFINE"}:
        bonus += 0.12
    return bonus


class AdaptiveQPolicy:
    def __init__(self, config: dict[str, Any], seed: int):
        self.config = config
        self.mode = str(config.get("mode", "train"))
        self.path = Path(str(config["artifact_path"])).resolve()
        self.rng = random.Random(seed)
        self.data = self._load()

    def _empty(self) -> dict[str, Any]:
        return {
            "artifact_version": POLICY_ARTIFACT_VERSION,
            "algorithm": "hierarchical_tabular_q_learning",
            "frozen": False,
            "created_at_utc": utc_now(), "updated_at_utc": utc_now(),
            "states": {}, "applied_updates": [],
        }

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            if self.mode == "frozen":
                raise FileNotFoundError(
                    f"Frozen policy artifact is missing: {self.path}. Complete development and freeze the policy first."
                )
            return self._empty()
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("artifact_version") != POLICY_ARTIFACT_VERSION:
            raise RuntimeError("Unsupported policy artifact version.")
        if self.mode == "frozen" and not bool(data.get("frozen")):
            raise RuntimeError("Pilot and holdout require an explicitly frozen policy artifact.")
        if self.mode == "train" and bool(data.get("frozen")):
            raise RuntimeError("A frozen policy cannot be updated. Use a new training artifact path.")
        return data

    def _stat(self, key: str, action: str) -> dict[str, Any]:
        return self.data.get("states", {}).get(key, {}).get(action, {})

    def estimated_q(self, key: str, action: str) -> tuple[float, int]:
        weighted = 0.0
        weight_total = 0.0
        visits = 0
        for parent, weight in _hierarchy(key):
            stat = self._stat(parent, action)
            count = int(stat.get("visits", 0))
            if count:
                confidence = count / (count + 3.0)
                weighted += weight * confidence * float(stat.get("q", 0.0))
                weight_total += weight * confidence
                visits += count
        return (weighted / weight_total if weight_total else 0.0), visits

    def rank_actions(self, key: str, state: MDPState, actions: list[str]) -> list[tuple[str, float]]:
        total_visits = sum(self.estimated_q(key, action)[1] for action in actions)
        exploration = float(self.config.get("ucb_exploration", 0.65))
        scores: list[tuple[str, float]] = []
        for action in actions:
            q_value, visits = self.estimated_q(key, action)
            score = q_value + _heuristic_bonus(state, action)
            if self.mode == "train":
                score += exploration * math.sqrt(math.log(total_visits + 2.0) / (visits + 1.0))
                score += self.rng.uniform(-1e-6, 1e-6)
            scores.append((action, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        if self.mode == "train" and self.rng.random() < float(self.config.get("epsilon", 0.18)):
            chosen = self.rng.randrange(len(scores))
            scores[0], scores[chosen] = scores[chosen], scores[0]
        return scores

    def update(
        self, *, update_id: str, key: str, action: str, reward: float,
        next_key: str, next_actions: list[str], terminal: bool, success: bool,
    ) -> None:
        if self.mode != "train":
            return
        applied = set(self.data.get("applied_updates", []))
        if update_id in applied:
            return
        next_q = 0.0
        if not terminal and next_actions:
            next_q = max(self.estimated_q(next_key, candidate)[0] for candidate in next_actions)
        target = float(reward) + float(self.config.get("discount", 0.82)) * next_q
        alpha = float(self.config.get("learning_rate", 0.32))
        states = self.data.setdefault("states", {})
        for parent, _ in _hierarchy(key):
            actions = states.setdefault(parent, {})
            stat = actions.setdefault(action, {"q": 0.0, "visits": 0, "successes": 0, "reward_sum": 0.0})
            stat["q"] = float(stat["q"]) + alpha * (target - float(stat["q"]))
            stat["visits"] = int(stat["visits"]) + 1
            stat["successes"] = int(stat["successes"]) + int(success)
            stat["reward_sum"] = float(stat["reward_sum"]) + float(reward)
        self.data.setdefault("applied_updates", []).append(update_id)
        self.data["updated_at_utc"] = utc_now()
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.data, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.path)

    def summary(self) -> dict[str, Any]:
        states = self.data.get("states", {})
        return {
            "mode": self.mode, "artifact_path": str(self.path),
            "frozen": bool(self.data.get("frozen")),
            "exact_states": len([key for key in states if "*" not in key]),
            "updates": len(self.data.get("applied_updates", [])),
        }

    def export_metrics(self, destination: str | Path) -> Path:
        rows: list[dict[str, Any]] = []
        for key, actions in self.data.get("states", {}).items():
            for action, stat in actions.items():
                visits = int(stat.get("visits", 0))
                rows.append({
                    "state_key": key, "action_id": action,
                    "q_value": float(stat.get("q", 0.0)), "visits": visits,
                    "successes": int(stat.get("successes", 0)),
                    "mean_reward": float(stat.get("reward_sum", 0.0)) / visits if visits else 0.0,
                })
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=[
            "state_key", "action_id", "q_value", "visits", "successes", "mean_reward",
        ]).to_csv(destination, index=False)
        return destination


def freeze_policy(training_path: str | Path, frozen_path: str | Path) -> dict[str, Any]:
    source = Path(training_path).resolve()
    destination = Path(frozen_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Training policy does not exist: {source}")
    data = json.loads(source.read_text(encoding="utf-8"))
    if not data.get("applied_updates"):
        raise RuntimeError("The training policy has no updates and cannot be frozen.")
    data["frozen"] = True
    data["frozen_at_utc"] = utc_now()
    data["source_training_sha256"] = sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, destination)
    return {
        "status": "frozen", "path": str(destination),
        "sha256": sha256_file(destination),
        "updates": len(data.get("applied_updates", [])),
    }
