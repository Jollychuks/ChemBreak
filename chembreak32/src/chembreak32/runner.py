from __future__ import annotations

import gc
import hashlib
import json
import os
import copy
import time
from pathlib import Path

from .checkpoint import Store
from .config import load_config, validate_config
from .constants import NAMESPACE, PACKAGE_VERSION, RESPONSE_CLASS_PROGRESS, SOURCE_PROMPTS_SHA256, MANIFEST_SHA256
from .dataset import selected_tasks
from .metrics import export_results
from .policy import HierarchicalPolicy
from .prompts import attack_prompt, candidate_judge_prompt, candidate_strategy, response_judge_prompt
from .providers import ChcsJudgeCascadeError, ProviderPolicyBlock, make_roles
from .reporting import Reporter
from .state import EpisodeState, failure_mode
from .targets import make_target
from .utils import config_fingerprint, max_text_similarity, write_json


class ChemBreak32Runner:
    def __init__(self, config_path, target_id):
        self.config = load_config(config_path)
        validate_config(self.config)
        self.target_id = str(target_id)
        targets = {str(item["id"]): item for item in self.config["targets"]}
        if self.target_id not in targets:
            raise ValueError(f"Unknown target {self.target_id}; valid={sorted(targets)}")
        self.target_cfg = targets[self.target_id]
        run = self.config["run"]
        experiment = self.config["experiment"]
        self.tasks = selected_tasks(run["prompts_path"], run["manifest_path"]).to_dict("records")
        limit = run.get("task_limit")
        self.tasks = self.tasks[: int(limit)] if limit else self.tasks
        self.out = Path(run["output_root"]) / run["experiment_revision"] / self.target_id
        self.out.mkdir(parents=True, exist_ok=True)
        self.store = Store(self.out / "state.sqlite3")
        self.art = Path(run["artifact_root"]) / run["experiment_revision"] / self.target_id
        self.art.mkdir(parents=True, exist_ok=True)
        self.active_task_id = None
        self.policy = None
        self.task_memory = None
        self.loaded = False
        roles = self.config["roles"]
        self.identity = {
            "namespace": NAMESPACE,
            "package_version": PACKAGE_VERSION,
            "experiment_revision": run["experiment_revision"],
            "run_mode": run["run_mode"],
            "run_id": run["run_id"],
            "config_sha256": config_fingerprint(self.config),
            "target_id": self.target_id,
            "target_model": self.target_cfg["model"],
            "target_revision": self.target_cfg["revision"],
            "source_prompts_sha256": SOURCE_PROMPTS_SHA256,
            "manifest_sha256": MANIFEST_SHA256,
            "assignment_ids": [str(item["assignment_id"]) for item in self.tasks],
            "seed": int(run["seed"]),
            "models": {
                "attack_llm": roles["attack_llm"]["model"],
                "intent_gate_llm": roles["intent_gate_llm"]["model"],
                "chcs_judge_llm": roles["chcs_judge_llm"]["model"],
                "chcs_fallback_judge_llm": roles["chcs_fallback_judge_llm"]["model"],
                "target": self.target_cfg["model"],
            },
            "experiment": dict(experiment),
            "candidate_gate": dict(self.config["candidate_gate"]),
            "judge_cascade": dict(self.config["judge_cascade"]),
            "chcs": dict(self.config["chcs"]),
            "policy": dict(self.config["policy"]),
            "reward": dict(self.config["reward"]),
            "technical_failures": dict(self.config["technical_failures"]),
            "final_recovery": dict(self.config["final_recovery"]),
            "task_isolation": {"cross_task_learning": False, "reset_before_each_task": True},
        }
        existing = self.store.get_meta("experiment_identity")
        if existing is not None and existing != self.identity:
            raise RuntimeError("Existing checkpoint belongs to another ChemBreak32 experiment identity. Use a new run_id.")
        self.store.set_meta("experiment_identity", self.identity)
        self.store.set_meta("task_count", len(self.tasks))
        active = self.store.get_meta("active_episode_checkpoint")
        if active:
            phase = str(active["phase"])
            episode = int(active["episode"])
            assignment_id = str(active["assignment_id"])
            if self.store.episode_status(phase, episode, assignment_id) != "complete":
                self.store.delete_episode_and_turns(phase, episode, assignment_id)
                self.store.set_meta("task_policy_snapshot", active.get("policy"))
                self.store.set_meta("task_memory_snapshot", active.get("memory"))
                self.store.set_meta("task_controller_assignment_id", assignment_id)
            self.store.set_meta("active_episode_checkpoint", None)
        self.store.clear_uncommitted_episodes()
        if not run["dry_run"] and self.store.contains_mock_markers():
            raise RuntimeError("Live run refused: this checkpoint contains mock records. Use a new live run_id.")
        project_id = None if run["dry_run"] else os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.roles = make_roles(self.config, project_id)
        self.target = make_target(self.target_cfg, run["dry_run"])
        self.reporter = Reporter((1 + int(experiment["adaptive_episodes"])) * len(self.tasks), enabled=run.get("live_progress", True))
        self.reporter.target_queries = len(self.store.target_queries())

    def _task_seed(self, assignment_id):
        payload = f"{self.config['run']['seed']}|{self.target_id}|{assignment_id}"
        return int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16)

    def _task_dir(self, assignment_id):
        path = self.art / str(assignment_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _empty_memory(assignment_id):
        return {
            "assignment_id": str(assignment_id),
            "baseline": None,
            "attempts": [],
            "latest_evaluated": None,
            "peak_chcs": 1,
            "best_attempt": None,
            "best_episode": None,
            "best_restart_action": "NONE",
            "episode_summaries": [],
            "unresolved_judgments": 0,
        }

    @staticmethod
    def _attempt_rank(item):
        return (
            int(bool(item.get("success"))),
            int(item.get("chcs") or 0),
            float(item.get("reward") or 0.0),
            -int(item.get("episode") or 0),
            -int(item.get("turn") or 0),
        )

    @classmethod
    def _refresh_best_strategy(cls, memory):
        judged = [item for item in memory.get("attempts", []) if item.get("chcs") is not None]
        memory["best_attempt"] = max(judged, key=cls._attempt_rank, default=None)
        episodes = list(memory.get("episode_summaries", []))
        memory["best_episode"] = max(
            episodes,
            key=lambda item: (
                int(bool(item.get("success"))),
                int(item.get("peak_chcs", 0)),
                float(item.get("cumulative_reward", 0.0)),
                -int(item.get("episode", 0)),
            ),
            default=None,
        )
        best_episode = memory.get("best_episode") or {}
        route = list(best_episode.get("route", []))
        peak_turn = int(best_episode.get("best_turn", 0) or 0)
        prefix = [item for item in route if int(item.get("turn", 0)) <= peak_turn] if peak_turn else route
        restart_action = next(
            (str(item.get("action")) for item in prefix if str(item.get("action")) != "CONTINUE_CONTEXT"),
            "NONE",
        )
        if restart_action == "NONE" and memory.get("best_attempt"):
            action = str(memory["best_attempt"].get("action", "NONE"))
            restart_action = action if action != "CONTINUE_CONTEXT" else "NONE"
        memory["best_restart_action"] = restart_action
        return memory

    @classmethod
    def _record_episode_strategy(cls, memory, episode, total_reward, success):
        attempts = [
            item for item in memory.get("attempts", [])
            if int(item.get("episode", 0)) == int(episode) and item.get("chcs") is not None
        ]
        if not attempts:
            return cls._refresh_best_strategy(memory)
        best = max(attempts, key=cls._attempt_rank)
        summary = {
            "episode": int(episode),
            "success": bool(success),
            "peak_chcs": max(int(item.get("chcs") or 0) for item in attempts),
            "best_turn": int(best.get("turn", 0)),
            "best_action": str(best.get("action", "NONE")),
            "cumulative_reward": float(total_reward),
            "route": [
                {
                    "turn": int(item.get("turn", 0)),
                    "action": str(item.get("action", "NONE")),
                    "chcs": int(item.get("chcs") or 0),
                    "response_class": str(item.get("response_class", "unknown")),
                    "reward": float(item.get("reward") or 0.0),
                    "candidate_excerpt": str(item.get("candidate_excerpt", ""))[-1200:],
                    "target_response_excerpt": str(item.get("target_response_excerpt", ""))[-1200:],
                }
                for item in attempts
            ],
        }
        retained = [item for item in memory.get("episode_summaries", []) if int(item.get("episode", 0)) != int(episode)]
        retained.append(summary)
        memory["episode_summaries"] = sorted(retained, key=lambda item: int(item.get("episode", 0)))
        return cls._refresh_best_strategy(memory)

    def _rebuild_memory(self, assignment_id):
        memory = self._empty_memory(assignment_id)
        turns = [item for item in self.store.turns() if str(item["assignment_id"]) == str(assignment_id)]
        for row in sorted(turns, key=lambda item: int(item.get("target_query_index", 0))):
            judge = json.loads(row.get("judge_json") or "{}")
            unresolved = bool(judge.get("judge_unresolved")) or judge.get("chcs") is None
            attempt = {
                "phase": row["phase"],
                "episode": int(row["episode"]),
                "turn": int(row["turn_index"]),
                "action": row["action_id"],
                "judge_status": "unresolved" if unresolved else "judged",
                "chcs": judge.get("chcs"),
                "response_class": judge.get("response_class", "unknown" if unresolved else ""),
                "failure_mode": failure_mode(judge.get("response_class", "unknown")),
                "reward": float(row.get("reward", 0.0)),
                "success": bool(judge.get("final_success", False)),
                "candidate_excerpt": str(row.get("prompt", ""))[-1200:],
                "target_response_excerpt": str(row.get("response", ""))[-1200:],
            }
            if row["phase"] == "baseline":
                memory["baseline"] = attempt
            else:
                memory["attempts"].append(attempt)
            if unresolved:
                memory["unresolved_judgments"] += 1
            else:
                memory["latest_evaluated"] = {
                    "chcs": int(judge["chcs"]),
                    "response_class": str(judge["response_class"]),
                    "action": row["action_id"],
                    "reward": float(row.get("reward", 0.0)),
                }
                memory["peak_chcs"] = max(memory["peak_chcs"], int(judge["chcs"]))
        for episode in sorted({int(item.get("episode", 0)) for item in memory["attempts"] if int(item.get("episode", 0)) > 0}):
            rows = [item for item in memory["attempts"] if int(item.get("episode", 0)) == episode]
            self._record_episode_strategy(
                memory,
                episode,
                sum(float(item.get("reward") or 0.0) for item in rows),
                any(bool(item.get("success")) for item in rows),
            )
        return self._refresh_best_strategy(memory)

    def _activate(self, assignment_id):
        assignment_id = str(assignment_id)
        if self.active_task_id == assignment_id:
            return
        checkpoint_task = self.store.get_meta("task_controller_assignment_id")
        seed = self._task_seed(assignment_id)
        policy_snapshot = self.store.get_meta("task_policy_snapshot") if checkpoint_task == assignment_id else None
        memory_snapshot = self.store.get_meta("task_memory_snapshot") if checkpoint_task == assignment_id else None
        self.policy = HierarchicalPolicy(self.config["policy"], seed, policy_snapshot)
        self.policy.bind_task(assignment_id)
        self.policy.metadata = self.identity | {"assignment_id": assignment_id, "scope": "task_local"}
        self.task_memory = memory_snapshot or self._rebuild_memory(assignment_id)
        self.active_task_id = assignment_id
        self.store.set_meta("task_controller_assignment_id", assignment_id)
        self.store.set_meta("task_policy_snapshot", self.policy.to_dict())
        self.store.set_meta("task_memory_snapshot", self.task_memory)
        write_json(
            self._task_dir(assignment_id) / "task_isolation_manifest.json",
            {
                "assignment_id": assignment_id,
                "target_id": self.target_id,
                "task_seed": seed,
                "policy_scope": "current_task_only",
                "memory_scope": "current_task_only",
                "fresh_target_context_each_episode": True,
                "cross_task_learning": False,
                "exact_route_replay": False,
                "trajectory_anchored_episode_restart": True,
                "candidate_pool_size": 3,
                "one_target_query_per_turn": True,
            },
        )

    def _deactivate(self, assignment_id, outcome):
        task_dir = self._task_dir(assignment_id)
        self.policy.save(task_dir / "final_task_policy.json")
        write_json(task_dir / "task_memory_INTERNAL.json", self.task_memory)
        write_json(task_dir / "task_outcome.json", outcome)
        self.store.set_meta("task_controller_assignment_id", None)
        self.store.set_meta("task_policy_snapshot", None)
        self.store.set_meta("task_memory_snapshot", None)
        self.active_task_id = None
        self.policy = None
        self.task_memory = None

    def load_target(self):
        if not self.loaded:
            print(f"[TARGET LOAD] {self.target_id} starting", flush=True)
            self.target.load()
            self.loaded = True
            print(f"[TARGET LOAD] {self.target_id} ready", flush=True)

    def unload_target(self):
        if self.loaded:
            print(f"[TARGET UNLOAD] {self.target_id} releasing model memory", flush=True)
            self.target.unload()
            self.loaded = False
            gc.collect()
            print(f"[TARGET UNLOAD] {self.target_id} complete", flush=True)

    def _gate(self, task, action, candidate, strategy_role="diversified_action_execution", roles=None):
        active_roles = roles or self.roles
        gate = active_roles.judge_candidate(
            candidate_judge_prompt(task, action, candidate, strategy_role, self.task_memory),
            candidate,
        )
        settings = self.config["candidate_gate"]
        gate["gate_pass"] = bool(
            not gate["drifted"]
            and float(gate["goal_preservation"]) >= float(settings["goal_preservation_min"])
            and float(gate["task_fidelity"]) >= float(settings["task_fidelity_min"])
            and float(gate["action_alignment"]) >= float(settings["action_alignment_min"])
            and float(gate["strategy_alignment"]) >= float(settings["strategy_alignment_min"])
        )
        return gate

    def _save_judge_role_calls(self, phase, episode, assignment_id, turn_index, judge_or_audit, total_latency):
        audit = dict(judge_or_audit or {})
        primary_status = str(audit.get("primary_judge_status", "valid"))
        self.store.save_role_call(
            phase,
            episode,
            assignment_id,
            turn_index=turn_index,
            role="chcs_primary_judge",
            provider=str(self.config["roles"]["chcs_judge_llm"]["provider"]),
            model=str(self.config["roles"]["chcs_judge_llm"]["model"]),
            status=primary_status,
            latency_seconds=float(audit.get("primary_latency_seconds", total_latency)),
            details={"error_type": audit.get("primary_judge_error_type", "")},
        )
        fallback_status = str(audit.get("fallback_judge_status", "not_called"))
        if fallback_status != "not_called" or bool(audit.get("fallback_used")):
            self.store.save_role_call(
                phase,
                episode,
                assignment_id,
                turn_index=turn_index,
                role="chcs_fallback_judge",
                provider=str(self.config["roles"]["chcs_fallback_judge_llm"]["provider"]),
                model=str(self.config["roles"]["chcs_fallback_judge_llm"]["model"]),
                status=fallback_status,
                latency_seconds=float(audit.get("fallback_latency_seconds", 0.0)),
                details={"error_type": audit.get("fallback_judge_error_type", "")},
            )

    def _judge(self, task, response, phase, episode, assignment_id, turn_index, roles=None):
        active_roles = roles or self.roles
        started = time.perf_counter()
        try:
            judge = active_roles.judge_response(response_judge_prompt(task, response), response)
        except ChcsJudgeCascadeError as exc:
            self._save_judge_role_calls(
                phase, episode, assignment_id, turn_index, exc.audit, time.perf_counter() - started
            )
            raise
        self._save_judge_role_calls(
            phase, episode, assignment_id, turn_index, judge, time.perf_counter() - started
        )
        score = int(self.config["chcs"]["success_score"])
        judge["harmfulness_normalized"] = (int(judge["chcs"]) - 1) / 4.0
        judge["final_success"] = int(judge["chcs"]) == score
        judge["success_rule"] = "first_valid_chcs_score"
        judge["independent_confirmation_used"] = False
        return judge

    def _reward(self, previous_chcs, task_peak, judge, previous_class):
        settings = self.config["reward"]
        chcs = int(judge["chcs"])
        value = float(settings["chcs_delta"]) * ((chcs - int(previous_chcs)) / 4.0)
        value += float(settings["behavioral_progress"]) * (
            (
                RESPONSE_CLASS_PROGRESS.get(str(judge["response_class"]), 0)
                - RESPONSE_CLASS_PROGRESS.get(str(previous_class), 0)
            )
            / 5.0
        )
        if chcs > int(task_peak):
            value += float(settings.get("new_peak_bonus", 0.0))
        if judge["final_success"]:
            value += float(settings["success_bonus"])
        if chcs == 1:
            value -= float(settings["chcs1_penalty"])
        if chcs == 4 and int(previous_chcs) == 4:
            value += float(settings.get("chcs4_hold_bonus", 0.0))
        if int(task_peak) >= 4 and chcs < int(previous_chcs):
            value -= float(settings.get("peak_regression_penalty", 0.0))
        no_progress = chcs <= int(previous_chcs) and RESPONSE_CLASS_PROGRESS.get(
            str(judge["response_class"]), 0
        ) <= RESPONSE_CLASS_PROGRESS.get(str(previous_class), 0)
        if no_progress and not judge["final_success"]:
            value -= float(settings["stagnation_penalty"])
        return round(value - float(settings["turn_cost"]), 6)

    def _candidate_slot(self, phase, episode, task, state, action, last_response, candidate_slot, pool_texts):
        assignment_id = str(task["assignment_id"])
        settings = self.config["candidate_gate"]
        max_attempts = max(1, int(settings["max_candidate_attempts_per_slot"]))
        feedback = None
        last_reason = ""
        strategy_stage, strategy_role, _ = candidate_strategy(state, self.task_memory, candidate_slot)
        for generation_attempt in range(1, max_attempts + 1):
            attack_started = time.perf_counter()
            try:
                data = self.roles.attack(
                    attack_prompt(
                        task,
                        action,
                        state,
                        last_response,
                        self.task_memory,
                        gate_feedback=feedback,
                        generation_attempt=generation_attempt,
                        candidate_slot=candidate_slot,
                    ),
                    action,
                )
                candidate = str(data["utterance"]).strip()
                last_reason = str(data.get("reason_code", ""))
                self.store.save_role_call(
                    phase,
                    episode,
                    assignment_id,
                    turn_index=state.turn_index + 1,
                    role="attack_llm",
                    provider=str(self.config["roles"]["attack_llm"]["provider"]),
                    model=str(self.config["roles"]["attack_llm"]["model"]),
                    status="valid",
                    latency_seconds=time.perf_counter() - attack_started,
                    candidate_slot=candidate_slot,
                    details={
                        "generation_attempt": generation_attempt,
                        "action": action,
                        "candidate_strategy_stage": strategy_stage,
                        "candidate_strategy_role": strategy_role,
                    },
                )
            except ProviderPolicyBlock as exc:
                self.store.save_role_call(
                    phase, episode, assignment_id,
                    turn_index=state.turn_index + 1,
                    role="attack_llm", provider=exc.provider,
                    model=str(self.config["roles"]["attack_llm"]["model"]),
                    status="policy_block", latency_seconds=time.perf_counter() - attack_started,
                    candidate_slot=candidate_slot,
                    details={
                        "generation_attempt": generation_attempt,
                        "action": action,
                        "error_code": exc.code,
                        "candidate_strategy_stage": strategy_stage,
                        "candidate_strategy_role": strategy_role,
                    },
                )
                self.store.save_provider_event(
                    phase,
                    episode,
                    assignment_id,
                    stage="attack_llm",
                    provider=exc.provider,
                    event_type="policy_block",
                    error_code=exc.code,
                    action_id=action,
                    message=exc.provider_message,
                )
                return None
            except Exception as exc:
                self.store.save_role_call(
                    phase, episode, assignment_id,
                    turn_index=state.turn_index + 1,
                    role="attack_llm", provider="vertex_genai",
                    model=str(self.config["roles"]["attack_llm"]["model"]),
                    status="error", latency_seconds=time.perf_counter() - attack_started,
                    candidate_slot=candidate_slot,
                    details={
                        "generation_attempt": generation_attempt,
                        "action": action,
                        "error_type": type(exc).__name__,
                        "candidate_strategy_stage": strategy_stage,
                        "candidate_strategy_role": strategy_role,
                    },
                )
                self.store.save_provider_event(
                    phase,
                    episode,
                    assignment_id,
                    stage="attack_llm",
                    provider="vertex_genai",
                    event_type="attack_llm_error",
                    error_code=type(exc).__name__,
                    action_id=action,
                    message=str(exc),
                )
                return None

            prior = [str(item.get("candidate_excerpt", "")) for item in self.task_memory.get("attempts", [])] + list(pool_texts)
            similarity = max_text_similarity(candidate, prior)
            threshold_key = "max_similarity_continue" if action == "CONTINUE_CONTEXT" else "max_similarity_non_continue"
            threshold = float(settings[threshold_key])
            if similarity > threshold:
                feedback = {
                    "type": "candidate_too_similar",
                    "max_similarity": similarity,
                    "threshold": threshold,
                }
                self.store.save_provider_event(
                    phase,
                    episode,
                    assignment_id,
                    stage="candidate_gate",
                    provider="local",
                    event_type="novelty_rejection",
                    error_code="candidate_too_similar",
                    action_id=action,
                    details=feedback | {"generation_attempt": generation_attempt, "candidate_slot": candidate_slot},
                )
                continue
            gate_started = time.perf_counter()
            try:
                gate = self._gate(task, action, candidate, strategy_role)
                self.store.save_role_call(
                    phase, episode, assignment_id,
                    turn_index=state.turn_index + 1,
                    role="intent_gate",
                    provider=str(self.config["roles"]["intent_gate_llm"]["provider"]),
                    model=str(self.config["roles"]["intent_gate_llm"]["model"]),
                    status="valid",
                    latency_seconds=time.perf_counter() - gate_started,
                    candidate_slot=candidate_slot,
                    details={
                        "generation_attempt": generation_attempt,
                        "action": action,
                        "candidate_strategy_stage": strategy_stage,
                        "candidate_strategy_role": strategy_role,
                    },
                )
            except Exception as exc:
                self.store.save_role_call(
                    phase, episode, assignment_id,
                    turn_index=state.turn_index + 1,
                    role="intent_gate", provider="vertex_genai",
                    model=str(self.config["roles"]["intent_gate_llm"]["model"]),
                    status="error", latency_seconds=time.perf_counter() - gate_started,
                    candidate_slot=candidate_slot,
                    details={"generation_attempt": generation_attempt, "action": action, "error_type": type(exc).__name__},
                )
                self.store.save_provider_event(
                    phase,
                    episode,
                    assignment_id,
                    stage="intent_gate",
                    provider="vertex_genai",
                    event_type="gate_error",
                    error_code=type(exc).__name__,
                    action_id=action,
                    message=str(exc),
                    details={"generation_attempt": generation_attempt, "candidate_slot": candidate_slot},
                )
                feedback = {"type": "technical_gate_failure", "instruction": "Generate a replacement for this candidate slot."}
                continue
            gate["max_prior_similarity"] = similarity
            gate["similarity_threshold"] = threshold
            gate["novelty_pass"] = True
            if gate["gate_pass"]:
                gate["attack_reason_code"] = last_reason
                gate["addressed_failure"] = str(data.get("addressed_failure", ""))
                gate["difference_from_prior"] = str(data.get("difference_from_prior", ""))
                gate["generation_attempt"] = generation_attempt
                gate["candidate_slot"] = candidate_slot
                gate["candidate_strategy_stage"] = strategy_stage
                gate["candidate_strategy_role"] = strategy_role
                weights = settings["ranking_weights"]
                novelty = max(0.0, 1.0 - similarity)
                gate["novelty_score"] = novelty
                gate["ranking_score"] = round(
                    float(weights["goal_preservation"]) * float(gate["goal_preservation"])
                    + float(weights["task_fidelity"]) * float(gate["task_fidelity"])
                    + float(weights["action_alignment"]) * float(gate["action_alignment"])
                    + float(weights["strategy_alignment"]) * float(gate["strategy_alignment"])
                    + float(weights["novelty"]) * novelty,
                    8,
                )
                return {
                    "candidate": candidate,
                    "gate": gate,
                    "reason": last_reason,
                    "slot": candidate_slot,
                    "strategy_stage": strategy_stage,
                    "strategy_role": strategy_role,
                }
            feedback = {
                "type": "candidate_gate_rejection",
                "reason_code": str(gate.get("reason_code", "drift")),
                "goal_preservation": float(gate["goal_preservation"]),
                "task_fidelity": float(gate["task_fidelity"]),
                "action_alignment": float(gate["action_alignment"]),
                "strategy_alignment": float(gate["strategy_alignment"]),
            }
            self.store.save_provider_event(
                phase,
                episode,
                assignment_id,
                stage="candidate_gate",
                provider="vertex_genai",
                event_type="goal_drift_rejection",
                error_code=str(gate.get("reason_code", "drift")),
                action_id=action,
                details=feedback | {"generation_attempt": generation_attempt, "candidate_slot": candidate_slot},
            )
        return None

    def _candidate_pool(self, phase, episode, task, state, action, last_response):
        requested = int(self.config["candidate_gate"]["candidates_per_turn"])
        pool = []
        pool_texts = []
        for candidate_slot in range(1, requested + 1):
            item = self._candidate_slot(
                phase, episode, task, state, action, last_response, candidate_slot, pool_texts
            )
            if item is not None:
                pool.append(item)
                pool_texts.append(item["candidate"])
        if not pool:
            return None, None, "", "candidate_pool_unavailable"
        selected = max(
            pool,
            key=lambda item: (float(item["gate"]["ranking_score"]), -int(item["slot"])),
        )
        summary = []
        for item in pool:
            gate = item["gate"]
            summary.append(
                {
                    "candidate_slot": int(item["slot"]),
                    "candidate_strategy_stage": str(item["strategy_stage"]),
                    "candidate_strategy_role": str(item["strategy_role"]),
                    "candidate_hash": hashlib.sha256(item["candidate"].encode()).hexdigest()[:16],
                    "ranking_score": float(gate["ranking_score"]),
                    "goal_preservation": float(gate["goal_preservation"]),
                    "task_fidelity": float(gate["task_fidelity"]),
                    "action_alignment": float(gate["action_alignment"]),
                    "strategy_alignment": float(gate["strategy_alignment"]),
                    "novelty_score": float(gate["novelty_score"]),
                    "selected": int(item["slot"]) == int(selected["slot"]),
                }
            )
        selected_gate = dict(selected["gate"])
        selected_gate.update(
            {
                "candidate_pool_requested": requested,
                "candidate_pool_valid": len(pool),
                "selected_candidate_slot": int(selected["slot"]),
                "selected_candidate_strategy_stage": str(selected["strategy_stage"]),
                "selected_candidate_strategy_role": str(selected["strategy_role"]),
                "candidate_pool_summary": summary,
                "three_candidate_selection": True,
            }
        )
        return selected["candidate"], selected_gate, selected["reason"], "ok"

    def _target_and_judge(self, phase, episode, task, state, context_id, action, candidate):
        assignment_id = str(task["assignment_id"])
        turn = state.turn_index + 1
        candidate_hash = hashlib.sha256(candidate.encode()).hexdigest()[:16]
        try:
            generation = self.target.generate(candidate, state.history)
            self.reporter.target_queries += 1
        except Exception as exc:
            self.store.save_provider_event(
                phase,
                episode,
                assignment_id,
                stage="target",
                provider=self.target_id,
                event_type="target_error",
                error_code=type(exc).__name__,
                action_id=action,
                message=str(exc),
            )
            return None, None, None, "target_error"
        query_index = self.store.save_target_query(
            phase=phase,
            episode=episode,
            assignment_id=assignment_id,
            context_id=context_id,
            turn_index=turn,
            action_id=action,
            candidate_hash=candidate_hash,
            prompt=candidate,
            response=generation.text,
            latency_seconds=generation.latency_seconds,
        )
        last_error = None
        attempts = 1 + int(self.config["technical_failures"].get("immediate_judge_recovery_attempts", 1))
        for judge_attempt in range(1, attempts + 1):
            try:
                judge = self._judge(
                    task,
                    generation.text,
                    phase,
                    episode,
                    assignment_id,
                    turn,
                )
                if judge_attempt > 1:
                    judge["immediate_recovery_attempt"] = judge_attempt - 1
                    self.store.save_provider_event(
                        phase,
                        episode,
                        assignment_id,
                        stage="chcs_immediate_recovery",
                        provider="primary_then_fallback",
                        event_type="recovered_judgment",
                        error_code="",
                        action_id=action,
                        details={"target_query_index": query_index, "recovery_attempt": judge_attempt - 1},
                    )
                self.store.finalize_target_query(query_index, judge, "judged")
                return generation, query_index, judge, "judged"
            except ChcsJudgeCascadeError as exc:
                last_error = exc
                if judge_attempt < attempts:
                    self.store.save_provider_event(
                        phase,
                        episode,
                        assignment_id,
                        stage="chcs_immediate_recovery",
                        provider="primary_then_fallback",
                        event_type="retry_after_both_judges_failed",
                        error_code="both_judges_failed",
                        action_id=action,
                        details={"target_query_index": query_index, "recovery_attempt": judge_attempt},
                    )
        self.store.mark_target_query_judge_error(query_index, str(last_error), last_error.audit)
        self.store.save_provider_event(
            phase,
            episode,
            assignment_id,
            stage="chcs_judge_cascade",
            provider="primary_then_fallback",
            event_type="unresolved_judgment",
            error_code="both_judges_failed",
            action_id=action,
            details={"target_query_index": query_index, "judge_provenance": last_error.audit},
        )
        judge = {
            "judge_unresolved": True,
            "judge_provenance": last_error.audit,
            "final_success": False,
            "response_class": "unknown",
            "chcs": None,
        }
        return generation, query_index, judge, "unresolved"

    def _save_turn(self, phase, episode, task, state, context_id, action, candidate, gate, generation, query_index, judge, reward, decision, reason):
        decision = dict(decision)
        decision["attack_reason_code"] = reason
        self.store.save_turn_with_learning(
            self.policy.to_dict(),
            self.task_memory,
            phase=phase,
            episode=episode,
            assignment_id=str(task["assignment_id"]),
            context_id=context_id,
            turn_index=state.turn_index,
            target_query_index=query_index,
            action_id=action,
            selection_mode=decision.get("mode", ""),
            state_key=decision.get("task_key", ""),
            q_task=decision.get("q_task_before", 0.0),
            candidate_hash=hashlib.sha256(candidate.encode()).hexdigest()[:16],
            candidate_source="attack_llm_three_candidate_ranked",
            candidate_gate=gate,
            prompt=candidate,
            response=generation.text,
            judge=judge,
            decision=decision,
            reward=reward,
            latency_seconds=generation.latency_seconds,
        )

    def run_baseline(self, task, index):
        assignment_id = str(task["assignment_id"])
        if self.store.episode_done("baseline", 0, assignment_id):
            return bool(self.store.episode_row("baseline", 0, assignment_id)["success"])
        self.reporter.start_task("baseline", 0, index, len(self.tasks), assignment_id)
        self.store.start_episode("baseline", 0, assignment_id)
        state = EpisodeState.initial(task, 0, 1)
        candidate = str(task["original_prompt"])
        generation, query_index, judge, status = self._target_and_judge(
            "baseline", 0, task, state, f"{assignment_id}:baseline", "ORIGINAL_TASK", candidate
        )
        if generation is None:
            self.store.complete_episode("baseline", 0, assignment_id, False, 0, 0.0, "target_error")
            return False
        if status == "judged":
            state.advance("ORIGINAL_TASK", candidate, generation.text, judge, 0.0)
            reason = "success" if judge["final_success"] else "baseline_complete"
        else:
            state.advance_unresolved("ORIGINAL_TASK", candidate, generation.text)
            reason = "baseline_unresolved"
        self.store.save_turn(
            phase="baseline",
            episode=0,
            assignment_id=assignment_id,
            context_id=f"{assignment_id}:baseline",
            turn_index=1,
            target_query_index=query_index,
            action_id="ORIGINAL_TASK",
            selection_mode="direct",
            state_key="baseline",
            q_task=0.0,
            candidate_hash=hashlib.sha256(candidate.encode()).hexdigest()[:16],
            candidate_source="original_task",
            candidate_gate={"gate_pass": True, "not_applicable": True},
            prompt=candidate,
            response=generation.text,
            judge=judge,
            decision={"mode": "direct_baseline", "learning_update_skipped": True},
            reward=0.0,
            latency_seconds=generation.latency_seconds,
        )
        success = bool(judge.get("final_success", False))
        self.store.complete_episode("baseline", 0, assignment_id, success, 1, 0.0, reason)
        self.reporter.complete("baseline", 0, index, len(self.tasks), assignment_id, success, 1, 0.0)
        return success

    def run_episode(self, task, index, episode):
        assignment_id = str(task["assignment_id"])
        if self.store.episode_done("adaptive", episode, assignment_id):
            return bool(self.store.episode_row("adaptive", episode, assignment_id)["success"])
        self._activate(assignment_id)
        epsilon = float(self.config["experiment"]["episode_epsilons"][episode - 1])
        max_turns = int(self.config["experiment"]["max_turns_per_episode"])
        state = EpisodeState.initial(task, episode, max_turns, self.task_memory)
        context_id = f"{assignment_id}:episode:{episode}"
        turn_exclusions = set()
        candidate_failures = 0
        max_candidate_failures = int(self.config["technical_failures"]["max_candidate_failures_per_turn"])
        last_response = ""
        total_reward = 0.0
        success = False
        completion_reason = "turn_budget_exhausted"
        self.reporter.start_task("adaptive", episode, index, len(self.tasks), assignment_id)
        self.store.set_meta(
            "active_episode_checkpoint",
            {
                "phase": "adaptive",
                "episode": episode,
                "assignment_id": assignment_id,
                "policy": self.policy.to_dict(),
                "memory": self.task_memory,
            },
        )
        self.store.start_episode("adaptive", episode, assignment_id)

        while state.turn_index < max_turns and not success:
            features = state.policy_features()
            decision = self.policy.select(
                assignment_id,
                features,
                epsilon,
                recent=state.decision_history,
                extra_blocked=turn_exclusions,
            )
            action = decision["action"]
            candidate, gate, reason, candidate_status = self._candidate_pool(
                "adaptive", episode, task, state, action, last_response
            )
            if candidate is None:
                candidate_failures += 1
                turn_exclusions.add(action)
                if candidate_failures >= max_candidate_failures:
                    completion_reason = "candidate_generation_unavailable"
                    break
                continue
            decision["candidate_pool_requested"] = int(gate.get("candidate_pool_requested", 3))
            decision["candidate_pool_valid"] = int(gate.get("candidate_pool_valid", 0))
            decision["selected_candidate_slot"] = int(gate.get("selected_candidate_slot", 0))
            decision["selected_candidate_score"] = float(gate.get("ranking_score", 0.0))
            decision["selected_candidate_strategy_stage"] = str(
                gate.get("selected_candidate_strategy_stage", "within_episode_adaptation")
            )
            decision["selected_candidate_strategy_role"] = str(
                gate.get("selected_candidate_strategy_role", "diversified_action_execution")
            )

            generation, query_index, judge, judge_status = self._target_and_judge(
                "adaptive", episode, task, state, context_id, action, candidate
            )
            if generation is None:
                candidate_failures += 1
                turn_exclusions.add(action)
                if candidate_failures >= max_candidate_failures:
                    completion_reason = "target_generation_unavailable"
                    break
                continue

            turn_exclusions.clear()
            candidate_failures = 0
            if judge_status == "judged":
                previous_chcs = state.chcs
                previous_class = state.response_class
                task_peak = int(self.task_memory.get("peak_chcs", 1))
                reward = self._reward(previous_chcs, task_peak, judge, previous_class)
                memory_item = state.advance(action, candidate, generation.text, judge, reward)
                success = bool(judge["final_success"])
                update = self.policy.update(
                    assignment_id,
                    features,
                    action,
                    reward,
                    state.policy_features(),
                    success or state.turn_index >= max_turns,
                )
                decision["update"] = update
            else:
                reward = float(self.config["technical_failures"]["unresolved_reward"])
                memory_item = state.advance_unresolved(
                    action,
                    candidate,
                    generation.text,
                    self.config["technical_failures"]["unresolved_response_class"],
                )
                decision["update"] = {"learning_update_skipped": True, "reason": "unresolved_judgment"}
                self.task_memory["unresolved_judgments"] += 1

            total_reward += reward
            attempt = {
                "phase": "adaptive",
                "episode": episode,
                "turn": state.turn_index,
                "action": action,
                "judge_status": judge_status,
                "chcs": judge.get("chcs"),
                "response_class": judge.get("response_class"),
                "failure_mode": failure_mode(judge.get("response_class", "unknown")),
                "reward": reward,
                "success": success,
                "candidate_excerpt": candidate[-1200:],
                "target_response_excerpt": generation.text[-1200:],
            }
            self.task_memory["attempts"].append(attempt)
            if judge_status == "judged":
                self.task_memory["latest_evaluated"] = {
                    "chcs": int(judge["chcs"]),
                    "response_class": str(judge["response_class"]),
                    "action": action,
                    "reward": reward,
                }
                self.task_memory["peak_chcs"] = max(int(self.task_memory["peak_chcs"]), int(judge["chcs"]))
                self._refresh_best_strategy(self.task_memory)
            decision["memory_item"] = memory_item
            self._save_turn(
                "adaptive",
                episode,
                task,
                state,
                context_id,
                action,
                candidate,
                gate,
                generation,
                query_index,
                judge,
                reward,
                decision,
                reason,
            )
            self.reporter.turn(
                "adaptive",
                episode,
                index,
                len(self.tasks),
                assignment_id,
                state.turn_index,
                action,
                judge,
                reward,
                decision,
                "attack_llm_three_candidate_ranked",
                len(turn_exclusions),
            )
            last_response = generation.text

        if success:
            completion_reason = "first_valid_chcs5_success"
        self._record_episode_strategy(self.task_memory, episode, total_reward, success)
        self.store.complete_episode_with_learning(
            "adaptive",
            episode,
            assignment_id,
            success,
            state.turn_index,
            total_reward,
            completion_reason,
            self.policy.to_dict(),
            self.task_memory,
        )
        self.store.set_meta("active_episode_checkpoint", None)
        task_dir = self._task_dir(assignment_id)
        self.policy.save(task_dir / f"policy_after_episode_{episode}.json")
        write_json(task_dir / f"memory_after_episode_{episode}_INTERNAL.json", self.task_memory)
        self.reporter.complete("adaptive", episode, index, len(self.tasks), assignment_id, success, state.turn_index, total_reward)
        return success

    def _task_completion_status(self, assignment_id):
        max_episode = int(self.config["experiment"]["adaptive_episodes"])
        baseline = self.store.episode_row("baseline", 0, assignment_id)
        if not baseline or baseline["status"] != "complete":
            return False, False, None
        if bool(baseline["success"]):
            return True, True, 0
        for episode in range(1, max_episode + 1):
            row = self.store.episode_row("adaptive", episode, assignment_id)
            if row and row["status"] == "complete" and bool(row["success"]):
                return True, True, episode
        final_row = self.store.episode_row("adaptive", max_episode, assignment_id)
        if final_row and final_row["status"] == "complete":
            return True, False, None
        return False, False, None

    def _all_tasks_complete(self):
        return all(self._task_completion_status(str(task["assignment_id"]))[0] for task in self.tasks)

    def _recovery_roles(self):
        recovery_config = copy.deepcopy(self.config)
        settings = self.config["final_recovery"]
        recovery_config["roles"]["chcs_judge_llm"]["attempts"] = int(settings["primary_attempts_per_query"])
        recovery_config["roles"]["chcs_judge_llm"]["timeout_seconds"] = float(settings["primary_timeout_seconds"])
        recovery_config["roles"]["chcs_judge_llm"]["retry_backoff_seconds"] = 0.0
        recovery_config["roles"]["chcs_fallback_judge_llm"]["attempts"] = int(settings["fallback_attempts_per_query"])
        recovery_config["roles"]["chcs_fallback_judge_llm"]["retry_backoff_seconds"] = 0.0
        project_id = None if self.config["run"]["dry_run"] else os.environ.get("GOOGLE_CLOUD_PROJECT")
        return make_roles(recovery_config, project_id)

    def recover_unresolved_and_export(self):
        tasks = {str(item["assignment_id"]): item for item in self.tasks}
        committed = self.store.committed_target_query_ids()
        settings = self.config["final_recovery"]
        pending = [item for item in self.store.unresolved_target_queries() if int(item["query_index"]) in committed]
        total_pending = len(pending)
        max_queries = int(settings["max_queries_per_target"])
        budget_seconds = float(settings["time_budget_seconds_per_target"])
        recovery_roles = self._recovery_roles()
        started = time.perf_counter()
        recovered = 0
        attempted = 0
        print(
            f"[FINAL RECOVERY] {self.target_id} pending={total_pending} max_queries={max_queries} time_budget_seconds={budget_seconds:.0f}",
            flush=True,
        )
        for query in pending:
            if attempted >= max_queries or time.perf_counter() - started >= budget_seconds:
                break
            attempted += 1
            assignment_id = str(query["assignment_id"])
            task = tasks.get(assignment_id)
            if task is None:
                continue
            print(
                f"[FINAL RECOVERY] {self.target_id} {attempted}/{min(total_pending, max_queries)} query={int(query['query_index'])} task={assignment_id}",
                flush=True,
            )
            try:
                judge = self._judge(
                    task,
                    str(query["response"]),
                    "recovery",
                    int(query["episode"]),
                    assignment_id,
                    int(query["turn_index"]),
                    roles=recovery_roles,
                )
                judge["recovery_pass"] = 1
                self.store.finalize_target_query(int(query["query_index"]), judge, "judged")
                recovered += 1
                if judge.get("final_success"):
                    self.store.mark_episode_recovered_success(
                        str(query["phase"]), int(query["episode"]), assignment_id
                    )
                self.store.save_provider_event(
                    "recovery", int(query["episode"]), assignment_id,
                    stage="chcs_recovery", provider="primary_then_fallback",
                    event_type="recovered_judgment", error_code="",
                    action_id=str(query["action_id"]),
                    details={"target_query_index": int(query["query_index"]), "final_success": bool(judge.get("final_success"))},
                )
            except ChcsJudgeCascadeError as exc:
                self.store.save_provider_event(
                    "recovery", int(query["episode"]), assignment_id,
                    stage="chcs_recovery", provider="primary_then_fallback",
                    event_type="recovery_failed", error_code=type(exc).__name__,
                    action_id=str(query["action_id"]), message=str(exc),
                    details={"target_query_index": int(query["query_index"])},
                )
        remaining = len([item for item in self.store.unresolved_target_queries() if int(item["query_index"]) in committed])
        elapsed = time.perf_counter() - started
        previous = int(self.store.get_meta("recovered_judgments", 0) or 0)
        self.store.set_meta("recovered_judgments", previous + recovered)
        self.store.set_meta(
            "final_recovery_status",
            {"pending_before": total_pending, "attempted": attempted, "recovered": recovered, "remaining": remaining, "elapsed_seconds": elapsed, "bounded": True},
        )
        print(
            f"[FINAL RECOVERY] {self.target_id} complete attempted={attempted} recovered={recovered} remaining={remaining} elapsed_seconds={elapsed:.1f}",
            flush=True,
        )
        return export_results(self.store, self.out / "release", self.target_id, self.config)

    def run_all(self):
        max_episodes = int(self.config["experiment"]["adaptive_episodes"])
        if self._all_tasks_complete():
            print(f"[RESUME] {self.target_id} target tasks already complete; exporting without loading the model", flush=True)
            return export_results(self.store, self.out / "release", self.target_id, self.config)
        self.load_target()
        try:
            for index, task in enumerate(self.tasks, 1):
                assignment_id = str(task["assignment_id"])
                task_complete, _, _ = self._task_completion_status(assignment_id)
                if task_complete:
                    continue
                baseline_success = self.run_baseline(task, index)
                if baseline_success:
                    write_json(
                        self._task_dir(assignment_id) / "task_outcome.json",
                        {
                            "assignment_id": assignment_id,
                            "success": True,
                            "success_phase": "baseline",
                            "success_episode": 0,
                        },
                    )
                    continue
                self._activate(assignment_id)
                success = False
                success_episode = None
                for episode in range(1, max_episodes + 1):
                    if self.run_episode(task, index, episode):
                        success = True
                        success_episode = episode
                        break
                self._deactivate(
                    assignment_id,
                    {
                        "assignment_id": assignment_id,
                        "success": success,
                        "success_phase": "adaptive" if success else "none",
                        "success_episode": success_episode,
                        "episodes_attempted": success_episode or max_episodes,
                    },
                )
            return export_results(self.store, self.out / "release", self.target_id, self.config)
        finally:
            self.unload_target()

    def close(self):
        self.store.close()
