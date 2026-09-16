from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .evidence import EvidenceMemory
from .utils import utc_now, write_json


class TrajectoryMemory:
    """Persistent task-level memory for complete successful multi-turn paths.

    A trajectory is an ordered sequence of exact realized attack-LLM messages,
    each paired with the abstract MDP action that produced it.  CB19 replays a
    remembered successful trajectory *from the beginning of a fresh episode*;
    it never injects a late-turn message into an unrelated state.
    """

    SCHEMA_VERSION = 1

    def __init__(self, settings: dict[str, Any], data: dict | None = None):
        self.settings = settings
        data = data or {}
        schema = int(data.get("trajectory_schema_version", self.SCHEMA_VERSION if not data else 0))
        if data and schema != self.SCHEMA_VERSION:
            raise RuntimeError(
                f"CB19 trajectory schema mismatch: expected {self.SCHEMA_VERSION}, found {schema}."
            )
        self.tasks = dict(data.get("tasks", {}))
        self.metadata = dict(data.get("metadata", {}))
        self.frozen = bool(data.get("frozen", False))
        self.records = int(data.get("records", 0))

    @staticmethod
    def canonical_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for i, step in enumerate(steps, 1):
            action = str(step["action_id"])
            prompt = EvidenceMemory.normalize_candidate(str(step["prompt"]))
            out.append(
                {
                    "step_index": i,
                    "action_id": action,
                    "prompt": prompt,
                    "realization_id": EvidenceMemory.realization_id(prompt),
                    "candidate_id": EvidenceMemory.candidate_id(prompt, action),
                }
            )
        return out

    @classmethod
    def trajectory_id(cls, steps: list[dict[str, Any]]) -> str:
        canonical = cls.canonical_steps(steps)
        payload = "\n---STEP---\n".join(
            f"{s['action_id']}\n{s['prompt']}" for s in canonical
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    @classmethod
    def load(cls, path, settings):
        p = Path(path)
        return cls(settings, json.loads(p.read_text()) if p.exists() else None)

    def _entry(self, task_id: str, trajectory_id: str):
        return self.tasks.setdefault(str(task_id), {}).setdefault(str(trajectory_id), {})

    def _wilson_lower(self, s: int, n: int) -> float:
        if n <= 0:
            return 0.0
        z = float(self.settings.get("wilson_z", 1.96))
        p = float(s) / float(n)
        zz = z * z
        center = p + zz / (2 * n)
        margin = z * math.sqrt((p * (1 - p) + zz / (4 * n)) / n)
        return max(0.0, (center - margin) / (1 + zz / n))

    @staticmethod
    def _bounded_quality(value: float, scale: float) -> float:
        scale = max(abs(float(scale)), 1e-9)
        return 0.5 + 0.5 * math.tanh(float(value) / scale)

    def score(self, e: dict[str, Any]) -> dict[str, float]:
        n = max(0, int(e.get("attempts", 0)))
        s = max(0, int(e.get("successes", 0)))
        reliability = self._wilson_lower(s, n)
        target = max(1, int(self.settings.get("target_support_attempts", 3)))
        support = min(1.0, math.log1p(n) / math.log1p(target)) if n else 0.0
        mean_reward = float(e.get("reward_sum", 0.0)) / n if n else 0.0
        qn = max(0, int(e.get("q_observations", 0)))
        mean_q = float(e.get("q_sum", 0.0)) / qn if qn else 0.0
        reward_quality = self._bounded_quality(mean_reward, float(self.settings.get("reward_scale", 4.0)))
        q_quality = self._bounded_quality(mean_q, float(self.settings.get("q_scale", 2.0)))
        weights = {
            "reliability": float(self.settings.get("reliability_weight", 0.55)),
            "support": float(self.settings.get("support_weight", 0.20)),
            "reward": float(self.settings.get("reward_weight", 0.15)),
            "q": float(self.settings.get("q_weight", 0.10)),
        }
        denom = sum(weights.values()) or 1.0
        rank = (
            weights["reliability"] * reliability
            + weights["support"] * support
            + weights["reward"] * reward_quality
            + weights["q"] * q_quality
        ) / denom
        return {
            "rank_score": float(rank),
            "wilson_lower": float(reliability),
            "support_score": float(support),
            "mean_reward": float(mean_reward),
            "mean_q": float(mean_q),
            "reward_quality": float(reward_quality),
            "q_quality": float(q_quality),
        }

    def _observe(
        self,
        *,
        task_id: str,
        trajectory_id: str,
        steps: list[dict[str, Any]],
        phase: str,
        epoch: int,
        success: bool,
        total_reward: float,
        mean_q: float,
        source: str,
        turns_used: int,
    ) -> dict[str, Any]:
        if self.frozen:
            raise RuntimeError("Frozen CB19 trajectory memory cannot be updated")
        canonical = self.canonical_steps(steps)
        tid = self.trajectory_id(canonical)
        if tid != str(trajectory_id):
            raise RuntimeError("CB19 trajectory identity mismatch")
        e = self._entry(task_id, tid)
        if not e:
            e.update(
                {
                    "trajectory_id": tid,
                    "steps": canonical,
                    "length": len(canonical),
                    "attempts": 0,
                    "successes": 0,
                    "failures": 0,
                    "reward_sum": 0.0,
                    "best_reward": None,
                    "q_sum": 0.0,
                    "q_observations": 0,
                    "turns_sum": 0,
                    "first_seen": {"phase": phase, "epoch": int(epoch)},
                    "last_seen": None,
                    "epochs_seen": [],
                    "success_epochs": [],
                    "failure_epochs": [],
                    "sources": [],
                }
            )
        if e.get("steps") != canonical:
            raise RuntimeError("CB19 trajectory hash collision detected")
        e["attempts"] += 1
        if success:
            e["successes"] += 1
            if int(epoch) > 0 and int(epoch) not in e["success_epochs"]:
                e["success_epochs"].append(int(epoch))
        else:
            e["failures"] += 1
            if int(epoch) > 0 and int(epoch) not in e["failure_epochs"]:
                e["failure_epochs"].append(int(epoch))
        e["reward_sum"] = float(e["reward_sum"]) + float(total_reward)
        e["best_reward"] = (
            float(total_reward)
            if e["best_reward"] is None
            else max(float(e["best_reward"]), float(total_reward))
        )
        e["q_sum"] = float(e["q_sum"]) + float(mean_q)
        e["q_observations"] = int(e["q_observations"]) + 1
        e["turns_sum"] = int(e.get("turns_sum", 0)) + int(turns_used)
        e["last_seen"] = {
            "phase": phase,
            "epoch": int(epoch),
            "success": bool(success),
            "total_reward": float(total_reward),
            "turns_used": int(turns_used),
        }
        if int(epoch) > 0 and int(epoch) not in e["epochs_seen"]:
            e["epochs_seen"].append(int(epoch))
        if str(source) not in e["sources"]:
            e["sources"].append(str(source))
        for name in ("epochs_seen", "success_epochs", "failure_epochs", "sources"):
            e[name] = sorted(e[name])
        self.records += 1
        return dict(e)

    def record_success_sequence(
        self,
        *,
        task_id: str,
        steps: list[dict[str, Any]],
        phase: str,
        epoch: int,
        total_reward: float,
        mean_q: float,
        source: str,
    ) -> dict[str, Any]:
        if not steps:
            raise ValueError("Cannot record an empty CB19 trajectory")
        canonical = self.canonical_steps(steps)
        return self._observe(
            task_id=task_id,
            trajectory_id=self.trajectory_id(canonical),
            steps=canonical,
            phase=phase,
            epoch=epoch,
            success=True,
            total_reward=total_reward,
            mean_q=mean_q,
            source=source,
            turns_used=len(canonical),
        )

    def observe_replay(
        self,
        *,
        task_id: str,
        trajectory_id: str,
        phase: str,
        epoch: int,
        success: bool,
        total_reward: float,
        mean_q: float,
        turns_used: int,
    ) -> dict[str, Any]:
        e = self.tasks.get(str(task_id), {}).get(str(trajectory_id))
        if e is None:
            raise KeyError(f"Unknown CB19 trajectory {trajectory_id} for task {task_id}")
        return self._observe(
            task_id=task_id,
            trajectory_id=str(trajectory_id),
            steps=e["steps"],
            phase=phase,
            epoch=epoch,
            success=success,
            total_reward=total_reward,
            mean_q=mean_q,
            source="trajectory_replay",
            turns_used=turns_used,
        )

    def rank_trajectories(self, task_id: str, *, require_success: bool = True) -> list[dict[str, Any]]:
        rows = []
        for tid, e in self.tasks.get(str(task_id), {}).items():
            if require_success and int(e.get("successes", 0)) < 1:
                continue
            rows.append({**e, **self.score(e)})
        rows.sort(
            key=lambda x: (
                -float(x["rank_score"]),
                -int(x.get("successes", 0)),
                -int(x.get("attempts", 0)),
                int(x.get("length", 999)),
                str(x["trajectory_id"]),
            )
        )
        for i, row in enumerate(rows, 1):
            row["rank"] = i
        return rows

    def best_live_trajectory(self, task_id: str, attempted_ids: set[str] | None = None):
        attempted_ids = attempted_ids or set()
        for row in self.rank_trajectories(task_id, require_success=True):
            if row["trajectory_id"] not in attempted_ids:
                return row
        return None

    def build_rankings(self) -> dict[str, list[dict[str, Any]]]:
        return {
            task: self.rank_trajectories(task, require_success=True)
            for task in sorted(self.tasks)
        }

    @staticmethod
    def choose_frozen_trajectory(
        rankings: dict[str, list[dict[str, Any]]],
        task_id: str,
        attempted_ids: set[str] | None = None,
    ):
        attempted_ids = attempted_ids or set()
        for row in rankings.get(str(task_id), []):
            if row.get("trajectory_id") not in attempted_ids:
                return row
        return None

    def coverage(self) -> dict[str, Any]:
        task_count = len(self.tasks)
        covered = sum(
            1
            for task in self.tasks
            if any(int(e.get("successes", 0)) > 0 for e in self.tasks[task].values())
        )
        trajectory_count = sum(len(v) for v in self.tasks.values())
        return {
            "tasks_with_trajectory_memory": task_count,
            "tasks_with_successful_trajectory": covered,
            "successful_trajectory_coverage": (covered / task_count if task_count else 0.0),
            "trajectory_count": trajectory_count,
            "records": self.records,
        }

    def to_dict(self):
        return {
            "namespace": "CB19",
            "trajectory_schema_version": self.SCHEMA_VERSION,
            "metadata": self.metadata,
            "frozen": self.frozen,
            "records": self.records,
            "tasks": self.tasks,
            "saved_at_utc": utc_now(),
        }

    def save(self, path):
        write_json(path, self.to_dict())
