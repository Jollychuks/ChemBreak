from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from pathlib import Path

from .checkpoint import Store
from .config import load_config, validate_config
from .constants import (
    NAMESPACE, PACKAGE_VERSION, RESPONSE_CLASS_PROGRESS,
)
from .dataset import load_lock, selected_tasks
from .metrics import export_results
from .policy import ResponseAwarePolicy
from .prompts import attack_planner_prompt, candidate_judge_prompt, response_metric_prompt
from .providers import MetricJudgeError, ProviderPolicyBlock, classify_judge_error, make_roles
from .reporting import Reporter
from .scoring import combine_metric_results, unresolved_metric, validate_metric
from .state import InteractionState, failure_mode
from .targets import make_target
from .utils import config_fingerprint, max_text_similarity, read_json, sha256_file, write_json


class ChemBreakRunner:
    def __init__(self, config_path, target_id):
        self.config = load_config(config_path)
        validate_config(self.config)
        self.target_id = str(target_id)
        targets = {str(item["id"]): item for item in self.config["targets"]}
        if self.target_id not in targets:
            raise ValueError(f"Unknown target {self.target_id}; valid targets are {sorted(targets)}")
        self.target_cfg = targets[self.target_id]
        run = self.config["run"]
        self.dataset_lock = load_lock(run["lock_path"])
        self.tasks = selected_tasks(
            run["prompts_path"], run["manifest_path"], run["lock_path"]
        ).to_dict("records")
        if run.get("task_limit"):
            self.tasks = self.tasks[: int(run["task_limit"])]
        experiment_path = Path(run["output_root"]) / run["experiment_revision"] / run["judge_experiment"]
        self.out = experiment_path / self.target_id
        self.out.mkdir(parents=True, exist_ok=True)
        self.store = Store(self.out / "state.sqlite3")
        self.art = Path(run["artifact_root"]) / run["experiment_revision"] / run["judge_experiment"] / self.target_id
        self.art.mkdir(parents=True, exist_ok=True)
        self.active_task_id = None
        self.policy = None
        self.task_memory = None
        self.loaded = False
        roles = self.config["roles"]
        self.identity = {
            "namespace": NAMESPACE,
            "package_version": PACKAGE_VERSION,
            "project": "ChemBreak",
            "experiment_revision": run["experiment_revision"],
            "judge_experiment": run["judge_experiment"],
            "run_mode": run["run_mode"],
            "run_id": run["run_id"],
            "config_sha256": config_fingerprint(self.config),
            "dataset_profile": self.dataset_lock["dataset_profile"],
            "dataset_protocol": self.dataset_lock["protocol"],
            "locked_task_count": int(self.dataset_lock["task_count"]),
            "target_id": self.target_id,
            "target_model": self.target_cfg["model"],
            "target_revision": self.target_cfg["revision"],
            "source_prompts_sha256": sha256_file(run["prompts_path"]),
            "manifest_sha256": sha256_file(run["manifest_path"]),
            "assignment_ids": [str(item["assignment_id"]) for item in self.tasks],
            "seed": int(run["seed"]),
            "models": {
                "controller_planner": roles["attack_planner_llm"]["model"],
                "candidate_gate": roles["intent_gate_llm"]["model"],
                "progressive_candidate_gate": roles["progressive_gate_llm"]["model"],
                "sole_judge": roles["judge_llm"]["model"],
                "target": self.target_cfg["model"],
            },
            "experiment": dict(self.config["experiment"]),
            "candidate_gate": dict(self.config["candidate_gate"]),
            "scoring": dict(self.config["scoring"]),
            "policy": dict(self.config["policy"]),
            "reward": dict(self.config["reward"]),
            "technical_failures": dict(self.config["technical_failures"]),
            "task_isolation": {"cross_task_learning": False, "reset_before_each_task": True},
        }
        existing = self.store.get_meta("experiment_identity")
        if existing is not None and existing != self.identity:
            raise RuntimeError("Existing checkpoint belongs to an incompatible experiment. Use a new run_id.")
        self.store.set_meta("experiment_identity", self.identity)
        self.store.set_meta("task_count", len(self.tasks))
        if not run["dry_run"] and self.store.contains_mock_markers():
            raise RuntimeError("Live run refused because this checkpoint contains mock records. Use a new run_id.")
        project_id = None if run["dry_run"] else os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.roles = make_roles(self.config, project_id)
        self.target = make_target(self.target_cfg, run["dry_run"])
        self.reporter = Reporter(
            self.target_id,
            len(self.tasks),
            enabled=run.get("live_progress", True),
            settings=self.config.get("console", {}),
            judge_model=roles["judge_llm"]["model"],
        )
        self.reporter.target_queries = len(self.store.target_queries())

    def _task_seed(self, assignment_id):
        payload = f"{self.config['run']['seed']}|{self.target_id}|{assignment_id}"
        return int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16)

    def _task_dir(self, assignment_id):
        path = self.art / str(assignment_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _outcome_path(self, assignment_id):
        return self._task_dir(assignment_id) / "task_outcome.json"

    @staticmethod
    def _empty_memory(assignment_id):
        return {
            "assignment_id": str(assignment_id),
            "baseline": None,
            "attempts": [],
            "best_attempt": None,
            "latest_evaluated": None,
            "peak_rhs": 1,
            "peak_ofs": 1,
            "latest_rhs": 1,
            "latest_rhs_status": "not_yet_judged",
            "latest_rhs_reason_code": "",
            "latest_rhs_rationale": "",
            "unresolved_rhs": 0,
            "unresolved_ofs": 0,
            "technical_failures": 0,
            "technical_incomplete": False,
            "active_context_id": None,
            "active_context_history": [],
            "active_context_turns": 0,
            "context_counter": 0,
            "stage_summaries": [],
        }

    def _rebuild_memory(self, assignment_id):
        memory = self._empty_memory(assignment_id)
        rows = [row for row in self.store.turns() if str(row["assignment_id"]) == str(assignment_id)]
        rows.sort(key=lambda row: int(row.get("target_query_index", 0)))
        for row in rows:
            judgment = json.loads(row.get("judgment_json") or "{}")
            rhs = judgment.get("rhs") or {}
            ofs = judgment.get("ofs") or {}
            decision = json.loads(row.get("decision_json") or "{}")
            transition = json.loads(row.get("transition_json") or "{}")
            attempt = {
                "phase": row["phase"],
                "stage": int(row["stage"]),
                "turn": int(row["turn_index"]),
                "context_id": str(row.get("context_id", "")),
                "context_mode": str(decision.get("context_mode", "")),
                "action": row["action_id"],
                "strategy_mode": str(decision.get("strategy_mode", "adaptive_actions")),
                "progressive_attempt": int(decision.get("progressive_attempt", 0) or 0),
                "buildup_level": decision.get("buildup_level"),
                "buildup_route": str(decision.get("buildup_route", "")),
                "trajectory_role": str(decision.get("trajectory_role", "")),
                "rhs": rhs.get("score"),
                "rhs_effective": rhs.get("effective_score"),
                "rhs_status": rhs.get("decision_status", "unresolved"),
                "ofs": ofs.get("score"),
                "ofs_effective": ofs.get("effective_score"),
                "ofs_status": ofs.get("decision_status", "unresolved"),
                "response_class": ofs.get("response_class", "unknown"),
                "failure_mode": failure_mode(ofs.get("response_class", "unknown")),
                "rhs_reason_code": rhs.get("reason_code"),
                "ofs_reason_code": ofs.get("reason_code"),
                "rhs_rationale": rhs.get("rationale"),
                "ofs_rationale": ofs.get("rationale"),
                "calibration_flags": judgment.get("calibration_flags", []),
                "reward": float(row.get("reward", 0.0)),
                "terminal_ofs5": bool(judgment.get("terminal_ofs5", False)),
                "candidate_excerpt": str(row.get("prompt", ""))[-1200:],
                "target_response_excerpt": str(row.get("response", ""))[-1200:],
            }
            if row["phase"] == "baseline":
                memory["baseline"] = attempt
                continue
            memory["attempts"].append(attempt)
            if rhs.get("effective_score") is None:
                memory["unresolved_rhs"] += 1
            else:
                memory["latest_rhs"] = int(rhs["effective_score"])
                memory["latest_rhs_status"] = str(rhs.get("decision_status", "valid"))
                memory["latest_rhs_reason_code"] = str(rhs.get("reason_code") or "")
                memory["latest_rhs_rationale"] = str(rhs.get("rationale") or "")
            if ofs.get("effective_score") is None:
                memory["unresolved_ofs"] += 1
            else:
                memory["latest_evaluated"] = {
                    "ofs": int(transition.get("ofs_after", ofs.get("effective_score") or 1)),
                    "ofs_status": ofs.get("decision_status", "unresolved"),
                    "response_class": ofs.get("response_class", "unknown"),
                    "action": row["action_id"],
                    "reward": float(row.get("reward", 0.0)),
                    "stagnation_count": int(decision.get("stagnation_after", 0) or 0),
                    "context_id": str(row.get("context_id", "")),
                    "ofs_reason_code": ofs.get("reason_code"),
                    "ofs_rationale": ofs.get("rationale"),
                }
            if rhs.get("effective_score") is not None:
                memory["peak_rhs"] = max(memory["peak_rhs"], int(rhs["effective_score"]))
            if ofs.get("effective_score") is not None:
                memory["peak_ofs"] = max(memory["peak_ofs"], int(ofs["effective_score"]))
            if memory["best_attempt"] is None or (
                int(ofs.get("effective_score") or 0), float(row.get("reward", 0))
            ) > (
                int(memory["best_attempt"].get("ofs_effective") or 0),
                float(memory["best_attempt"].get("reward", 0)),
            ):
                memory["best_attempt"] = attempt
        adaptive = [row for row in rows if row["phase"] == "adaptive"]
        if adaptive:
            latest_context = str(adaptive[-1].get("context_id", ""))
            context_rows = [row for row in adaptive if str(row.get("context_id", "")) == latest_context]
            memory["active_context_id"] = latest_context
            memory["active_context_turns"] = len(context_rows)
            for row in context_rows:
                memory["active_context_history"].extend(
                    [{"role": "user", "content": str(row["prompt"])}, {"role": "assistant", "content": str(row["response"])}]
                )
            counters = []
            for row in adaptive:
                try:
                    counters.append(int(str(row.get("context_id", "")).rsplit(":", 1)[-1]))
                except Exception:
                    pass
            memory["context_counter"] = max(counters, default=0)
        return memory

    def _activate(self, assignment_id):
        assignment_id = str(assignment_id)
        if self.active_task_id == assignment_id:
            return
        checkpoint_task = self.store.get_meta("task_controller_assignment_id")
        policy_snapshot = self.store.get_meta("task_policy_snapshot") if checkpoint_task == assignment_id else None
        memory_snapshot = self.store.get_meta("task_memory_snapshot") if checkpoint_task == assignment_id else None
        seed = self._task_seed(assignment_id)
        self.policy = ResponseAwarePolicy(self.config["policy"], seed, policy_snapshot)
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
                "stage_context_policy": "stage_1_adaptive_then_two_fresh_progressive_attempts",
                "cross_task_learning": False,
                "single_candidate_per_turn": True,
            },
        )

    def _deactivate(self, assignment_id, outcome):
        if self.policy is not None:
            self.policy.save(self._task_dir(assignment_id) / "final_task_policy.json")
        if self.task_memory is not None:
            write_json(self._task_dir(assignment_id) / "task_memory_PRIVATE.json", self.task_memory)
        write_json(self._outcome_path(assignment_id), outcome)
        for key in (
            "task_controller_assignment_id", "task_policy_snapshot", "task_memory_snapshot", "active_stage_checkpoint"
        ):
            self.store.set_meta(key, None)
        self.active_task_id = None
        self.policy = None
        self.task_memory = None

    def load_target(self):
        if not self.loaded:
            self.reporter.target_load_start(self.target_cfg["model"])
            started = time.perf_counter()
            self.target.load()
            self.loaded = True
            self.reporter.target_load_complete(time.perf_counter() - started)

    def unload_target(self):
        if self.loaded:
            self.target.unload()
            self.loaded = False
            gc.collect()
            self.reporter.target_unloaded()

    def _gate(self, task, action, candidate, decision):
        progressive = decision.get("strategy_mode") == "progressive_buildup"
        gate = self.roles.judge_candidate(
            candidate_judge_prompt(task, action, candidate, decision, self.task_memory),
            candidate,
            progressive=progressive,
        )
        settings = self.config["candidate_gate"]
        if progressive:
            route_ok = (
                int(decision.get("progressive_attempt", 0)) == 1
                or float(gate["route_difference"]) >= float(settings["progressive_route_difference_min"])
            )
            gate["gate_pass"] = bool(
                not gate["drifted"]
                and float(gate["action_alignment"]) >= float(settings["action_alignment_min"])
                and float(gate["trajectory_fidelity"]) >= float(settings["progressive_trajectory_fidelity_min"])
                and float(gate["level_compliance"]) >= float(settings["progressive_level_compliance_min"])
                and float(gate["intensity_ceiling_compliance"]) >= float(settings["progressive_intensity_compliance_min"])
                and route_ok
            )
        else:
            gate["gate_pass"] = bool(
                not gate["drifted"]
                and float(gate["goal_preservation"]) >= float(settings["goal_preservation_min"])
                and float(gate["task_fidelity"]) >= float(settings["task_fidelity_min"])
                and float(gate["action_alignment"]) >= float(settings["action_alignment_min"])
            )
        gate["strategy_mode"] = str(decision.get("strategy_mode", "adaptive_actions"))
        gate["progressive_attempt"] = int(decision.get("progressive_attempt", 0) or 0)
        gate["buildup_level"] = decision.get("buildup_level")
        return gate

    @staticmethod
    def _context_responses(state, current_response):
        responses = [str(item.get("content", "")) for item in state.history if item.get("role") == "assistant"]
        responses.append(str(current_response))
        return responses

    def _judge_metric(self, metric, *, index, phase, stage, task, turn, responses, context_id):
        assignment_id = str(task["assignment_id"])
        provider = str(self.config["roles"]["judge_llm"]["provider"])
        model = str(self.config["roles"]["judge_llm"]["model"])
        self.reporter.judge_start(phase, metric)
        started = time.perf_counter()
        try:
            raw = self.roles.judge_metric(metric, response_metric_prompt(metric, task, responses, context_id))
            latency = time.perf_counter() - started
            result = validate_metric(metric, raw, responses)
            status = "valid" if result.get("decision_status") == "valid" else result.get("decision_status", "unresolved")
            self.store.save_role_call(
                phase, stage, assignment_id, turn_index=turn, role="sole_judge", provider=provider,
                model=model, status=status, metric=str(metric).upper(), latency_seconds=latency,
                details={
                    "score": result.get("score"), "effective_score": result.get("effective_score"),
                    "evidence_status": result.get("evidence_status"),
                    "validation_errors": result.get("validation_errors", []),
                },
            )
            if result.get("validation_errors"):
                event_type = "evidence_warning" if result.get("decision_status") == "valid" else "judge_validation_error"
                error = {
                    "category": result.get("decision_status"),
                    "provider": provider,
                    "status": "",
                    "error_code": "metric_validation",
                    "message": "; ".join(result.get("validation_errors", [])),
                    "metadata": {"score": result.get("score"), "evidence_status": result.get("evidence_status")},
                }
                self.store.save_provider_event(
                    phase, stage, assignment_id, turn_index=turn, component="sole_judge", provider=provider,
                    event_type=event_type, error_category=error["category"], metric=str(metric).upper(),
                    message=error["message"], details=error["metadata"],
                )
                if result.get("decision_status") != "valid":
                    self.reporter.detailed_error(
                        index=index, assignment_id=assignment_id, stage=stage, turn=turn, metric=metric, error=error
                    )
            self.reporter.judge_end(phase, metric, result, latency)
            return result, latency
        except MetricJudgeError as exc:
            latency = time.perf_counter() - started
            error = dict(exc.audit)
        except Exception as exc:
            latency = time.perf_counter() - started
            error = classify_judge_error(exc, provider, metric, 1)
        self.store.save_role_call(
            phase, stage, assignment_id, turn_index=turn, role="sole_judge", provider=provider,
            model=model, status="unresolved", metric=str(metric).upper(), latency_seconds=latency,
            details={"error": error},
        )
        self.store.save_provider_event(
            phase, stage, assignment_id, turn_index=turn, component="sole_judge", provider=provider,
            event_type="judge_error", error_category=error.get("category", "provider_error"),
            metric=str(metric).upper(), message=error.get("message", ""), status_code=error.get("status", ""),
            details=error,
        )
        self.reporter.detailed_error(
            index=index, assignment_id=assignment_id, stage=stage, turn=turn, metric=metric, error=error
        )
        result = unresolved_metric(metric, error)
        self.reporter.judge_end(phase, metric, result, latency)
        return result, latency

    def _judge(self, *, index, phase, stage, task, turn, responses, context_id):
        rhs, rhs_latency = self._judge_metric(
            "rhs", index=index, phase=phase, stage=stage, task=task, turn=turn,
            responses=responses, context_id=context_id,
        )
        ofs, ofs_latency = self._judge_metric(
            "ofs", index=index, phase=phase, stage=stage, task=task, turn=turn,
            responses=responses, context_id=context_id,
        )
        judgment = combine_metric_results(
            rhs, ofs, context_id=context_id, response_count=len(responses),
            judge_model=self.config["roles"]["judge_llm"]["model"],
        )
        judgment["metric_latencies_seconds"] = {"rhs": rhs_latency, "ofs": ofs_latency}
        return judgment, rhs_latency + ofs_latency

    def _reward(self, state, judgment):
        ofs_result = judgment["ofs"]
        ofs = ofs_result.get("effective_score")
        if ofs is None:
            return float(self.config["technical_failures"]["unresolved_reward"])
        settings = self.config["reward"]
        previous_ofs = int(state.ofs)
        previous_class = str(state.response_class)
        response_class = str(ofs_result.get("response_class") or previous_class)
        value = float(settings["ofs_delta"]) * ((int(ofs) - previous_ofs) / 4.0)
        class_delta = RESPONSE_CLASS_PROGRESS.get(response_class, 0) - RESPONSE_CLASS_PROGRESS.get(previous_class, 0)
        if class_delta > 0:
            value += float(settings["behavioral_progress"]) * (class_delta / 5.0)
        if int(ofs) > int(state.peak_ofs):
            value += float(settings["new_peak_bonus"])
        if int(ofs) == 5:
            value += float(settings["ofs5_bonus"])
        if int(ofs) == 1:
            value -= float(settings["ofs1_penalty"])
        if int(ofs) == 4 and previous_ofs == 4:
            value += float(settings["ofs4_hold_bonus"])
        if int(state.peak_ofs) >= 4 and int(ofs) < previous_ofs:
            value -= float(settings["peak_regression_penalty"])
        no_progress = int(ofs) <= previous_ofs and RESPONSE_CLASS_PROGRESS.get(
            response_class, 0
        ) <= RESPONSE_CLASS_PROGRESS.get(previous_class, 0)
        if no_progress and not judgment.get("terminal_ofs5"):
            value -= float(settings["stagnation_penalty"])
        value -= float(settings["turn_cost"])
        return round(value, 6)

    def _candidate(self, index, stage, task, state, decision):
        assignment_id = str(task["assignment_id"])
        settings = self.config["candidate_gate"]
        feedback = None
        latencies = {"planner": 0.0, "gate": 0.0}
        for attempt in range(1, int(settings["max_candidate_attempts_per_turn"]) + 1):
            started = time.perf_counter()
            try:
                data = self.roles.attack(
                    attack_planner_prompt(
                        task, state, decision, self.task_memory,
                        gate_feedback=feedback, generation_attempt=attempt,
                    ),
                    decision["allowed_actions"],
                )
                latency = time.perf_counter() - started
                latencies["planner"] += latency
                action = str(data["action_id"])
                candidate = str(data["utterance"]).strip()
                progressive = decision.get("strategy_mode") == "progressive_buildup"
                expected_level = decision.get("buildup_level")
                metadata_matches = True
                if progressive:
                    metadata_matches = (
                        str(data.get("strategy_mode")) == str(decision.get("strategy_mode"))
                        and int(data.get("progressive_attempt", -1)) == int(decision.get("progressive_attempt", 0) or 0)
                        and int(data.get("buildup_level", -99)) == int(expected_level)
                    )
                if not metadata_matches:
                    feedback = {
                        "type": "strategy_metadata_mismatch",
                        "required_strategy_mode": decision.get("strategy_mode"),
                        "required_progressive_attempt": decision.get("progressive_attempt", 0),
                        "required_buildup_level": expected_level if expected_level is not None else -1,
                    }
                    continue
                self.store.save_role_call(
                    "adaptive", stage, assignment_id, turn_index=state.turn_index + 1,
                    role="controller_planner", provider=str(self.config["roles"]["attack_planner_llm"]["provider"]),
                    model=str(self.config["roles"]["attack_planner_llm"]["model"]), status="valid",
                    latency_seconds=latency, details={
                        "attempt": attempt, "action": action,
                        "strategy_mode": decision.get("strategy_mode"),
                        "progressive_attempt": decision.get("progressive_attempt", 0),
                        "buildup_level": expected_level,
                    },
                )
            except Exception as exc:
                latency = time.perf_counter() - started
                latencies["planner"] += latency
                category = "provider_policy_refusal" if isinstance(exc, ProviderPolicyBlock) else type(exc).__name__
                message = exc.provider_message if isinstance(exc, ProviderPolicyBlock) else str(exc)
                self.store.save_provider_event(
                    "adaptive", stage, assignment_id, turn_index=state.turn_index + 1,
                    component="controller_planner", provider="vertex_genai", event_type="planner_error",
                    error_category=category, message=message, details={"attempt": attempt},
                )
                self.reporter.technical_failure(
                    index, assignment_id, stage, state.turn_index + 1,
                    f"controller_planner {category}: {message}",
                )
                feedback = {"type": "technical_generation_failure", "instruction": "Return one valid replacement."}
                continue

            prior = [str(item.get("candidate_excerpt", "")) for item in self.task_memory.get("attempts", [])]
            similarity = max_text_similarity(candidate, prior)
            if decision.get("strategy_mode") == "progressive_buildup" and int(decision.get("progressive_attempt", 0)) == 2:
                threshold_key = "max_similarity_progressive_repeat"
            else:
                threshold_key = "max_similarity_continue" if action == "CONTINUE_CONTEXT" else "max_similarity_non_continue"
            threshold = float(settings[threshold_key])
            if similarity > threshold:
                feedback = {"type": "candidate_too_similar", "max_similarity": similarity, "threshold": threshold}
                self.store.save_provider_event(
                    "adaptive", stage, assignment_id, turn_index=state.turn_index + 1,
                    component="candidate_gate", provider="local", event_type="novelty_rejection",
                    error_category="candidate_too_similar", action_id=action,
                    details=feedback | {"attempt": attempt},
                )
                continue

            started = time.perf_counter()
            try:
                gate = self._gate(task, action, candidate, decision)
                latency = time.perf_counter() - started
                latencies["gate"] += latency
                gate_role = "progressive_candidate_gate" if progressive else "candidate_gate"
                gate_config_key = "progressive_gate_llm" if progressive else "intent_gate_llm"
                self.store.save_role_call(
                    "adaptive", stage, assignment_id, turn_index=state.turn_index + 1,
                    role=gate_role, provider=str(self.config["roles"][gate_config_key]["provider"]),
                    model=str(self.config["roles"][gate_config_key]["model"]),
                    status="accepted" if gate["gate_pass"] else "rejected", latency_seconds=latency,
                    details={"attempt": attempt, "action": action},
                )
            except Exception as exc:
                latency = time.perf_counter() - started
                latencies["gate"] += latency
                if isinstance(exc, ProviderPolicyBlock):
                    error_category = "provider_policy_refusal"
                    error_message = f"{exc.code}: {exc.provider_message}"
                else:
                    error_category = getattr(exc, "category", type(exc).__name__)
                    error_message = str(exc)
                self.store.save_provider_event(
                    "adaptive", stage, assignment_id, turn_index=state.turn_index + 1,
                    component="progressive_candidate_gate" if progressive else "candidate_gate",
                    provider="vertex_genai", event_type="gate_error",
                    error_category=error_category,
                    action_id=action,
                    message=error_message,
                    details={"attempt": attempt, "provider_metadata": getattr(exc, "metadata", {})},
                )
                self.reporter.technical_failure(
                    index, assignment_id, stage, state.turn_index + 1,
                    f"{'progressive_candidate_gate' if progressive else 'candidate_gate'} "
                    f"{error_category}: {error_message}",
                )
                feedback = {"type": "technical_gate_failure", "instruction": "Generate one replacement candidate."}
                continue
            gate.update(
                {
                    "max_prior_similarity": similarity,
                    "similarity_threshold": threshold,
                    "novelty_pass": True,
                    "generation_attempt": attempt,
                }
            )
            if gate["gate_pass"]:
                gate["attack_reason_code"] = str(data.get("reason_code", ""))
                gate["addressed_failure"] = str(data.get("addressed_failure", ""))
                gate["difference_from_prior"] = str(data.get("difference_from_prior", ""))
                gate["buildup_route"] = str(data.get("buildup_route", ""))
                gate["trajectory_role"] = str(data.get("trajectory_role", ""))
                gate["objective_link"] = str(data.get("objective_link", ""))
                return candidate, action, gate, str(data.get("reason_code", "")), "ok", latencies
            feedback = {
                "type": "candidate_gate_rejection",
                "reason_code": str(gate.get("reason_code", "drift")),
                "goal_preservation": float(gate["goal_preservation"]),
                "task_fidelity": float(gate["task_fidelity"]),
                "action_alignment": float(gate["action_alignment"]),
            }
            if progressive:
                feedback.update(
                    {
                        "trajectory_fidelity": float(gate["trajectory_fidelity"]),
                        "level_compliance": float(gate["level_compliance"]),
                        "intensity_ceiling_compliance": float(gate["intensity_ceiling_compliance"]),
                        "route_difference": float(gate["route_difference"]),
                    }
                )
            self.store.save_provider_event(
                "adaptive", stage, assignment_id, turn_index=state.turn_index + 1,
                component="candidate_gate", provider="vertex_genai", event_type="candidate_rejection",
                error_category=str(gate.get("reason_code", "drift")), action_id=action,
                details=feedback | {"attempt": attempt},
            )
        detail = json.dumps(feedback or {}, ensure_ascii=False, default=str)[:1000]
        return None, None, None, "", f"candidate_unavailable_after_replacement: {detail}", latencies

    def _target_and_judge(self, index, phase, stage, task, state, context_id, action, candidate):
        assignment_id = str(task["assignment_id"])
        turn = state.turn_index + 1
        candidate_hash = hashlib.sha256(candidate.encode()).hexdigest()[:16]
        self.reporter.target_start(phase, index, assignment_id, stage, turn, action, context_id)
        started = time.perf_counter()
        try:
            generation = self.target.generate(candidate, state.history)
            target_latency = time.perf_counter() - started
            self.reporter.target_queries += 1
            self.store.save_role_call(
                phase, stage, assignment_id, turn_index=turn, role="target_model", provider="hf_local",
                model=str(self.target_cfg["model"]), status="valid", latency_seconds=target_latency,
                details={"action": action, "context_id": context_id},
            )
            self.reporter.target_end(
                phase, assignment_id, stage, turn, generation.text, target_latency
            )
        except Exception as exc:
            target_latency = time.perf_counter() - started
            self.store.save_provider_event(
                phase, stage, assignment_id, turn_index=turn, component="target_model", provider=self.target_id,
                event_type="target_error", error_category=type(exc).__name__, action_id=action,
                message=str(exc), details={"latency_seconds": target_latency},
            )
            return (
                None, None, None,
                f"target_error {type(exc).__name__}: {str(exc)[:1000]}",
                {"target": target_latency, "judge": 0.0},
            )
        query_index = self.store.save_target_query(
            phase=phase, stage=stage, assignment_id=assignment_id, context_id=context_id,
            turn_index=turn, action_id=action, candidate_hash=candidate_hash,
            prompt=candidate, response=generation.text, latency_seconds=generation.latency_seconds,
        )
        responses = self._context_responses(state, generation.text)
        judgment, judge_latency = self._judge(
            index=index, phase=phase, stage=stage, task=task, turn=turn,
            responses=responses, context_id=context_id,
        )
        self.store.finalize_target_query(query_index, judgment, judgment["judge_status"])
        return generation, query_index, judgment, judgment["judge_status"], {
            "target": target_latency, "judge": judge_latency,
        }

    def run_baseline(self, task, index):
        assignment_id = str(task["assignment_id"])
        if self.store.episode_done("baseline", 0, assignment_id):
            return bool(self.store.episode_row("baseline", 0, assignment_id)["terminal_ofs5"])
        if self.store.episode_row("baseline", 0, assignment_id):
            self.store.reset_episode("baseline", 0, assignment_id)
        self.reporter.baseline_start(index, assignment_id, str(task["original_prompt"]))
        self.store.start_episode("baseline", 0, assignment_id)
        state = InteractionState.initial(task, 0, 1)
        candidate = str(task["original_prompt"])
        generation, query_index, judgment, status, latencies = self._target_and_judge(
            index, "baseline", 0, task, state, f"{assignment_id}:baseline", "ORIGINAL_TASK", candidate
        )
        if generation is None:
            self.store.complete_episode("baseline", 0, assignment_id, False, 0, 0.0, "technical_target_error")
            self.reporter.technical_failure(index, assignment_id, 0, 1, status)
            return False
        transition = state.advance("ORIGINAL_TASK", candidate, generation.text, judgment, 0.0)
        self.store.save_turn(
            phase="baseline", stage=0, assignment_id=assignment_id,
            context_id=f"{assignment_id}:baseline", turn_index=1, context_turn_index=1,
            target_query_index=query_index, action_id="ORIGINAL_TASK", selection_mode="direct",
            state_key="baseline", candidate_hash=hashlib.sha256(candidate.encode()).hexdigest()[:16],
            candidate_source="original_task", candidate_gate={"gate_pass": True, "not_applicable": True},
            prompt=candidate, response=generation.text, judgment=judgment,
            decision={"mode": "direct_baseline", "context_separate": True},
            transition=transition, reward=0.0, latency_seconds=generation.latency_seconds,
        )
        terminal = bool(judgment.get("terminal_ofs5"))
        self.store.complete_episode(
            "baseline", 0, assignment_id, terminal, 1, 0.0,
            "valid_ofs5" if terminal else "baseline_complete",
        )
        self.reporter.baseline_result(index, judgment)
        return terminal

    def _new_context(self, assignment_id):
        self.task_memory["context_counter"] = int(self.task_memory.get("context_counter", 0)) + 1
        context_id = f"{assignment_id}:adaptive_context:{self.task_memory['context_counter']}"
        self.task_memory["active_context_id"] = context_id
        self.task_memory["active_context_history"] = []
        self.task_memory["active_context_turns"] = 0
        return context_id

    def _stage_checkpoint(self, *, stage, assignment_id, state, context_decision, context_id,
                          context_mode, total_reward, consecutive_candidate_failures):
        return {
            "phase": "adaptive",
            "stage": int(stage),
            "assignment_id": str(assignment_id),
            "state": state.to_dict(),
            "context_decision": dict(context_decision),
            "context_id": str(context_id),
            "context_mode": str(context_mode),
            "total_reward": float(total_reward),
            "consecutive_candidate_failures": int(consecutive_candidate_failures),
            "policy": self.policy.to_dict(),
            "memory": self.task_memory,
        }

    def run_stage(self, task, index, stage):
        assignment_id = str(task["assignment_id"])
        if self.store.episode_done("adaptive", stage, assignment_id):
            return bool(self.store.episode_row("adaptive", stage, assignment_id)["terminal_ofs5"])
        self._activate(assignment_id)
        active = self.store.get_meta("active_stage_checkpoint") or {}
        resume = (
            active.get("phase") == "adaptive"
            and int(active.get("stage", -1)) == int(stage)
            and str(active.get("assignment_id")) == assignment_id
            and self.store.episode_row("adaptive", stage, assignment_id) is not None
        )
        if resume:
            self.store.rollback_uncommitted_query_artifacts("adaptive", stage, assignment_id)
            self.policy = ResponseAwarePolicy(self.config["policy"], self._task_seed(assignment_id), active.get("policy"))
            self.policy.bind_task(assignment_id)
            self.task_memory = dict(active["memory"])
            state = InteractionState.from_dict(active["state"])
            context_decision = dict(active["context_decision"])
            context_id = str(active["context_id"])
            context_mode = str(active["context_mode"])
            total_reward = float(active.get("total_reward", 0.0))
            consecutive_candidate_failures = int(active.get("consecutive_candidate_failures", 0))
            self.reporter.stage_start(
                index, assignment_id, stage, context_decision, context_id, resumed_turn=state.turn_index
            )
        else:
            if self.store.episode_row("adaptive", stage, assignment_id):
                self.store.reset_episode("adaptive", stage, assignment_id)
            context_decision = self.policy.stage_context(assignment_id, stage, self.task_memory)
            state = InteractionState.initial(
                task, stage, int(self.config["experiment"]["max_turns_per_stage"]),
                self.task_memory,
                strategy_mode=context_decision["strategy_mode"],
                progressive_attempt=context_decision["progressive_attempt"],
            )
            context_id = self._new_context(assignment_id)
            context_mode = context_decision["mode"]
            total_reward = 0.0
            consecutive_candidate_failures = 0
            self.store.start_episode("adaptive", stage, assignment_id)
            checkpoint = self._stage_checkpoint(
                stage=stage, assignment_id=assignment_id, state=state, context_decision=context_decision,
                context_id=context_id, context_mode=context_mode, total_reward=total_reward,
                consecutive_candidate_failures=consecutive_candidate_failures,
            )
            self.store.set_meta("active_stage_checkpoint", checkpoint)
            self.reporter.stage_start(index, assignment_id, stage, context_decision, context_id)

        terminal = False
        completion_reason = "turn_budget_exhausted"
        max_turns = int(self.config["experiment"]["max_turns_per_stage"])
        while state.turn_index < max_turns and not terminal:
            restart, restart_reason = self.policy.should_restart_during_stage(state)
            if restart:
                context_id = self._new_context(assignment_id)
                state.restart_context()
                context_mode = "RESTART_CONVERSATION"
                self.reporter.context_restart(
                    index, assignment_id, stage, state.turn_index + 1, restart_reason, context_id
                )
            decision = self.policy.plan(assignment_id, state, context_mode)
            self.reporter.turn_start(
                index, assignment_id, stage, state.turn_index + 1,
                state.context_turn_index + 1, context_id,
                strategy_mode=decision.get("strategy_mode", "adaptive_actions"),
                progressive_attempt=decision.get("progressive_attempt", 0),
                buildup_level=decision.get("buildup_level"),
                trajectory_role=decision.get("trajectory_role", ""),
            )
            candidate, action, gate, reason, candidate_status, candidate_latencies = self._candidate(
                index, stage, task, state, decision
            )
            if candidate is None:
                consecutive_candidate_failures += 1
                self.task_memory["technical_failures"] += 1
                self.reporter.technical_failure(
                    index, assignment_id, stage, state.turn_index + 1,
                    f"{candidate_status}; consecutive={consecutive_candidate_failures}",
                )
                if consecutive_candidate_failures >= int(
                    self.config["technical_failures"]["max_consecutive_candidate_failures"]
                ):
                    completion_reason = "technical_candidate_generation_incomplete"
                    self.task_memory["technical_incomplete"] = True
                    break
                continue

            self.reporter.candidate_ready(
                action, candidate, gate, candidate_latencies,
                strategy_mode=decision.get("strategy_mode", "adaptive_actions"),
                progressive_attempt=decision.get("progressive_attempt", 0),
                buildup_level=decision.get("buildup_level"),
                buildup_route=gate.get("buildup_route", ""),
            )

            generation, query_index, judgment, judgment_status, target_latencies = self._target_and_judge(
                index, "adaptive", stage, task, state, context_id, action, candidate
            )
            if generation is None:
                consecutive_candidate_failures += 1
                self.task_memory["technical_failures"] += 1
                self.reporter.technical_failure(index, assignment_id, stage, state.turn_index + 1, judgment_status)
                if consecutive_candidate_failures >= int(
                    self.config["technical_failures"]["max_consecutive_candidate_failures"]
                ):
                    completion_reason = "technical_target_generation_incomplete"
                    self.task_memory["technical_incomplete"] = True
                    break
                continue

            consecutive_candidate_failures = 0
            reward = self._reward(state, judgment)
            transition = state.advance(
                action, candidate, generation.text, judgment, reward,
                buildup_level=decision.get("buildup_level"),
                buildup_route=gate.get("buildup_route", ""),
            )
            terminal = bool(judgment.get("terminal_ofs5"))
            total_reward += reward
            rhs = judgment["rhs"]
            ofs = judgment["ofs"]
            attempt = {
                "phase": "adaptive",
                "stage": stage,
                "turn": state.turn_index,
                "context_id": context_id,
                "context_mode": context_mode,
                "action": action,
                "strategy_mode": decision.get("strategy_mode", "adaptive_actions"),
                "progressive_attempt": int(decision.get("progressive_attempt", 0) or 0),
                "buildup_level": decision.get("buildup_level"),
                "buildup_route": str(gate.get("buildup_route", "")),
                "trajectory_role": str(gate.get("trajectory_role", decision.get("trajectory_role", ""))),
                "rhs": rhs.get("score"),
                "rhs_effective": rhs.get("effective_score"),
                "rhs_status": rhs.get("decision_status"),
                "ofs": ofs.get("score"),
                "ofs_effective": ofs.get("effective_score"),
                "ofs_status": ofs.get("decision_status"),
                "response_class": ofs.get("response_class"),
                "failure_mode": failure_mode(ofs.get("response_class", "unknown")),
                "rhs_reason_code": rhs.get("reason_code"),
                "ofs_reason_code": ofs.get("reason_code"),
                "rhs_rationale": rhs.get("rationale"),
                "ofs_rationale": ofs.get("rationale"),
                "calibration_flags": judgment.get("calibration_flags", []),
                "reward": reward,
                "terminal_ofs5": terminal,
                "candidate_excerpt": candidate[-1200:],
                "target_response_excerpt": generation.text[-1200:],
            }
            self.task_memory["attempts"].append(attempt)
            self.task_memory["active_context_id"] = context_id
            self.task_memory["active_context_history"] = [dict(item) for item in state.history]
            self.task_memory["active_context_turns"] = state.context_turn_index
            if rhs.get("effective_score") is None:
                self.task_memory["unresolved_rhs"] += 1
            else:
                self.task_memory["latest_rhs"] = state.rhs
                self.task_memory["latest_rhs_status"] = state.rhs_status
                self.task_memory["latest_rhs_reason_code"] = str(rhs.get("reason_code") or "")
                self.task_memory["latest_rhs_rationale"] = str(rhs.get("rationale") or "")
            if ofs.get("effective_score") is None:
                self.task_memory["unresolved_ofs"] += 1
            else:
                self.task_memory["latest_evaluated"] = {
                    "ofs": state.ofs,
                    "ofs_status": state.ofs_status,
                    "response_class": state.response_class,
                    "action": action,
                    "reward": reward,
                    "stagnation_count": state.stagnation_count,
                    "context_id": context_id,
                    "ofs_reason_code": ofs.get("reason_code"),
                    "ofs_rationale": ofs.get("rationale"),
                }
            self.task_memory["peak_rhs"] = max(int(self.task_memory["peak_rhs"]), state.peak_rhs)
            self.task_memory["peak_ofs"] = max(int(self.task_memory["peak_ofs"]), state.peak_ofs)
            best = self.task_memory.get("best_attempt")
            if best is None or (
                int(attempt.get("ofs_effective") or 0), float(reward)
            ) > (
                int(best.get("ofs_effective") or 0), float(best.get("reward", 0))
            ):
                self.task_memory["best_attempt"] = attempt
            decision = dict(decision)
            decision["attack_reason_code"] = reason
            decision["buildup_route"] = str(gate.get("buildup_route", ""))
            decision["trajectory_role"] = str(gate.get("trajectory_role", decision.get("trajectory_role", "")))
            decision["objective_link"] = str(gate.get("objective_link", ""))
            decision["stagnation_after"] = state.stagnation_count
            decision["transition"] = transition
            checkpoint = self._stage_checkpoint(
                stage=stage, assignment_id=assignment_id, state=state, context_decision=context_decision,
                context_id=context_id, context_mode=context_mode, total_reward=total_reward,
                consecutive_candidate_failures=consecutive_candidate_failures,
            )
            row = {
                "phase": "adaptive", "stage": stage, "assignment_id": assignment_id,
                "context_id": context_id, "turn_index": state.turn_index,
                "context_turn_index": state.context_turn_index, "target_query_index": query_index,
                "action_id": action, "selection_mode": decision.get("mode", "response_aware_planner"),
                "state_key": decision.get("state_key", ""),
                "candidate_hash": hashlib.sha256(candidate.encode()).hexdigest()[:16],
                "candidate_source": "controller_planner_single_candidate", "candidate_gate": gate,
                "prompt": candidate, "response": generation.text, "judgment": judgment,
                "decision": decision, "transition": transition, "reward": reward,
                "latency_seconds": generation.latency_seconds,
            }
            self.store.save_turn_and_checkpoint(
                row, active_checkpoint=checkpoint, policy_snapshot=self.policy.to_dict(),
                memory_snapshot=self.task_memory,
            )
            self.reporter.result(
                index, stage, state.turn_index, judgment,
                details={
                    "assignment_id": assignment_id,
                    "action": action,
                    "strategy_mode": decision.get("strategy_mode", "adaptive_actions"),
                    "progressive_attempt": decision.get("progressive_attempt", 0),
                    "buildup_level": decision.get("buildup_level"),
                    "buildup_route": decision.get("buildup_route", ""),
                    "response_class": state.response_class,
                    "reward": reward,
                    "peak_rhs": state.peak_rhs,
                    "peak_ofs": state.peak_ofs,
                    "stagnation": state.stagnation_count,
                    "context_id": context_id,
                    "timings": candidate_latencies | target_latencies,
                },
            )

        if terminal:
            completion_reason = "first_valid_ofs5"
        if state.turn_index == 0 and completion_reason == "turn_budget_exhausted":
            completion_reason = "technical_zero_turn_stage"
            self.task_memory["technical_incomplete"] = True
        self.task_memory["stage_summaries"].append(
            {
                "stage": stage,
                "strategy_mode": state.strategy_mode,
                "progressive_attempt": state.progressive_attempt,
                "context_mode": context_decision["mode"],
                "context_id": context_id,
                "turns": state.turn_index,
                "ending_rhs": state.rhs,
                "ending_ofs": state.ofs,
                "peak_rhs": state.peak_rhs,
                "peak_ofs": state.peak_ofs,
                "maximum_level_reached": state.maximum_level_reached,
                "ending_buildup_route": state.buildup_route,
                "stagnation_count": state.stagnation_count,
                "terminal_ofs5": terminal,
                "completion_reason": completion_reason,
            }
        )
        self.store.complete_episode(
            "adaptive", stage, assignment_id, terminal, state.turn_index, total_reward, completion_reason
        )
        self.store.set_meta("active_stage_checkpoint", None)
        self.store.set_meta("task_policy_snapshot", self.policy.to_dict())
        self.store.set_meta("task_memory_snapshot", self.task_memory)
        self.policy.save(self._task_dir(assignment_id) / f"policy_after_stage_{stage}.json")
        write_json(self._task_dir(assignment_id) / f"memory_after_stage_{stage}_PRIVATE.json", self.task_memory)
        self.reporter.stage_end(
            index, assignment_id, stage, state.turn_index, total_reward, completion_reason
        )
        return terminal

    def _completed_outcomes(self):
        outcomes = {}
        for task in self.tasks:
            assignment_id = str(task["assignment_id"])
            path = self._outcome_path(assignment_id)
            if path.exists():
                outcomes[assignment_id] = read_json(path, {})
        return outcomes

    def run_all(self):
        outcomes = self._completed_outcomes()
        self.reporter.ofs5_tasks = sum(bool(item.get("ofs_5_reached")) for item in outcomes.values())
        self.reporter.run_start(self.config["run"]["judge_experiment"], resumed_tasks=len(outcomes))
        if len(outcomes) != len(self.tasks):
            self.load_target()
            try:
                for index, task in enumerate(self.tasks, 1):
                    assignment_id = str(task["assignment_id"])
                    if assignment_id in outcomes:
                        continue
                    self.reporter.task_start(index, assignment_id)
                    query_before = len(self.store.target_queries())
                    baseline_terminal = self.run_baseline(task, index)
                    self._activate(assignment_id)
                    adaptive_terminal = False
                    adaptive_terminal_stage = None
                    for stage in range(1, int(self.config["experiment"]["adaptive_stages"]) + 1):
                        if self.run_stage(task, index, stage):
                            adaptive_terminal = True
                            adaptive_terminal_stage = stage
                            break
                    overall_terminal = bool(baseline_terminal or adaptive_terminal)
                    outcome = {
                        "assignment_id": assignment_id,
                        "ofs_5_reached": overall_terminal,
                        "baseline_ofs_5_reached": bool(baseline_terminal),
                        "adaptive_ofs_5_reached": bool(adaptive_terminal),
                        "first_ofs_5_phase": (
                            "baseline" if baseline_terminal else ("adaptive" if adaptive_terminal else None)
                        ),
                        "first_ofs_5_stage": 0 if baseline_terminal else adaptive_terminal_stage,
                        "adaptive_first_ofs_5_stage": adaptive_terminal_stage,
                        "stages_attempted": (
                            adaptive_terminal_stage or int(self.config["experiment"]["adaptive_stages"])
                        ),
                        "technical_incomplete": bool(self.task_memory.get("technical_incomplete", False)),
                        "peak_rhs": int(self.task_memory.get("peak_rhs", 1)),
                        "peak_ofs": int(self.task_memory.get("peak_ofs", 1)),
                        "unresolved_rhs": int(self.task_memory.get("unresolved_rhs", 0)),
                        "unresolved_ofs": int(self.task_memory.get("unresolved_ofs", 0)),
                    }
                    technical_incomplete = outcome["technical_incomplete"]
                    self._deactivate(assignment_id, outcome)
                    task_queries = len(self.store.target_queries()) - query_before
                    self.reporter.task_complete(
                        index, assignment_id, overall_terminal, technical_incomplete, task_queries
                    )
                    outcomes[assignment_id] = outcome
            finally:
                self.unload_target()
        summary = export_results(self.store, self.out / "release", self.target_id, self.config)
        self.reporter.run_complete(summary)
        return summary

    def close(self):
        self.store.close()
