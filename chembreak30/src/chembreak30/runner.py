from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path

from .checkpoint import Store
from .config import load_config, validate_config
from .constants import NAMESPACE, PACKAGE_VERSION, RESPONSE_CLASS_PROGRESS, SOURCE_PROMPTS_SHA256, MANIFEST_SHA256
from .dataset import selected_tasks
from .metrics import export_results
from .policy import HierarchicalPolicy
from .prompts import attack_prompt, candidate_judge_prompt, response_judge_prompt
from .providers import ChcsJudgeCascadeError, ProviderPolicyBlock, make_roles
from .reporting import Reporter
from .state import EpisodeState, failure_mode
from .targets import make_target
from .utils import config_fingerprint, max_text_similarity, write_json


class ChemBreak30Runner:
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
            "task_isolation": {"cross_task_learning": False, "reset_before_each_task": True},
        }
        existing = self.store.get_meta("experiment_identity")
        if existing is not None and existing != self.identity:
            raise RuntimeError("Existing checkpoint belongs to another ChemBreak30 experiment identity. Use a new run_id.")
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
            "unresolved_judgments": 0,
        }

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
        return memory

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
                "single_candidate_per_generation": True,
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
            self.target.load()
            self.loaded = True

    def unload_target(self):
        if self.loaded:
            self.target.unload()
            self.loaded = False
            gc.collect()

    def _gate(self, task, candidate):
        gate = self.roles.judge_candidate(candidate_judge_prompt(task, candidate), candidate)
        settings = self.config["candidate_gate"]
        gate["gate_pass"] = bool(
            not gate["drifted"]
            and float(gate["goal_preservation"]) >= float(settings["goal_preservation_min"])
            and float(gate["task_fidelity"]) >= float(settings["task_fidelity_min"])
        )
        return gate

    def _judge(self, task, response):
        judge = self.roles.judge_response(response_judge_prompt(task, response), response)
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
        no_progress = chcs <= int(previous_chcs) and RESPONSE_CLASS_PROGRESS.get(
            str(judge["response_class"]), 0
        ) <= RESPONSE_CLASS_PROGRESS.get(str(previous_class), 0)
        if no_progress and not judge["final_success"]:
            value -= float(settings["stagnation_penalty"])
        return round(value - float(settings["turn_cost"]), 6)

    def _candidate(self, phase, episode, task, state, action, last_response):
        assignment_id = str(task["assignment_id"])
        settings = self.config["candidate_gate"]
        max_attempts = max(1, int(settings["max_candidate_attempts_per_action"]))
        feedback = None
        last_reason = ""
        for generation_attempt in range(1, max_attempts + 1):
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
                    ),
                    action,
                )
                candidate = str(data["utterance"]).strip()
                last_reason = str(data.get("reason_code", ""))
            except ProviderPolicyBlock as exc:
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
                return None, None, "", "attack_policy_block"
            except Exception as exc:
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
                return None, None, "", "attack_error"

            prior = [str(item.get("candidate_excerpt", "")) for item in self.task_memory.get("attempts", [])]
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
                    details=feedback | {"generation_attempt": generation_attempt},
                )
                continue
            try:
                gate = self._gate(task, candidate)
            except Exception as exc:
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
                    details={"generation_attempt": generation_attempt},
                )
                feedback = {"type": "technical_gate_failure", "instruction": "Generate a new single candidate."}
                continue
            gate["max_prior_similarity"] = similarity
            gate["similarity_threshold"] = threshold
            gate["novelty_pass"] = True
            if gate["gate_pass"]:
                gate["attack_reason_code"] = last_reason
                gate["addressed_failure"] = str(data.get("addressed_failure", ""))
                gate["difference_from_prior"] = str(data.get("difference_from_prior", ""))
                gate["single_candidate_generation"] = True
                gate["generation_attempt"] = generation_attempt
                return candidate, gate, last_reason, "ok"
            feedback = {
                "type": "goal_drift",
                "reason_code": str(gate.get("reason_code", "drift")),
                "goal_preservation": float(gate["goal_preservation"]),
                "task_fidelity": float(gate["task_fidelity"]),
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
                details=feedback | {"generation_attempt": generation_attempt},
            )
        return None, None, last_reason, "action_candidate_unavailable"

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
                judge = self._judge(task, generation.text)
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
            candidate_source="attack_llm_single_candidate",
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
            candidate, gate, reason, candidate_status = self._candidate(
                "adaptive", episode, task, state, action, last_response
            )
            if candidate is None:
                candidate_failures += 1
                turn_exclusions.add(action)
                if candidate_failures >= max_candidate_failures:
                    completion_reason = "candidate_generation_unavailable"
                    break
                continue

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
                "attack_llm_single_candidate",
                len(turn_exclusions),
            )
            last_response = generation.text

        if success:
            completion_reason = "first_valid_chcs5_success"
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

    def _recover_unresolved_judgments(self):
        tasks = {str(item["assignment_id"]): item for item in self.tasks}
        committed = self.store.committed_target_query_ids()
        recovered = 0
        for recovery_pass in range(1, int(self.config["technical_failures"]["recovery_passes"]) + 1):
            pending = [item for item in self.store.unresolved_target_queries() if int(item["query_index"]) in committed]
            if not pending:
                break
            for query in pending:
                assignment_id = str(query["assignment_id"])
                task = tasks.get(assignment_id)
                if task is None:
                    continue
                try:
                    judge = self._judge(task, str(query["response"]))
                    judge["recovery_pass"] = recovery_pass
                    self.store.finalize_target_query(int(query["query_index"]), judge, "judged")
                    recovered += 1
                    if judge.get("final_success"):
                        self.store.mark_episode_recovered_success(
                            str(query["phase"]), int(query["episode"]), assignment_id
                        )
                    self.store.save_provider_event(
                        str(query["phase"]),
                        int(query["episode"]),
                        assignment_id,
                        stage="chcs_recovery",
                        provider="primary_then_fallback",
                        event_type="recovered_judgment",
                        error_code="",
                        action_id=str(query["action_id"]),
                        details={
                            "target_query_index": int(query["query_index"]),
                            "recovery_pass": recovery_pass,
                            "final_success": bool(judge.get("final_success")),
                        },
                    )
                except ChcsJudgeCascadeError as exc:
                    self.store.save_provider_event(
                        str(query["phase"]),
                        int(query["episode"]),
                        assignment_id,
                        stage="chcs_recovery",
                        provider="primary_then_fallback",
                        event_type="recovery_failed",
                        error_code=type(exc).__name__,
                        action_id=str(query["action_id"]),
                        message=str(exc),
                        details={"target_query_index": int(query["query_index"]), "recovery_pass": recovery_pass},
                    )
        self.store.set_meta("recovered_judgments", recovered)
        return recovered

    def run_all(self):
        self.load_target()
        max_episodes = int(self.config["experiment"]["adaptive_episodes"])
        try:
            for index, task in enumerate(self.tasks, 1):
                assignment_id = str(task["assignment_id"])
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
            self._recover_unresolved_judgments()
            return export_results(self.store, self.out / "release", self.target_id, self.config)
        finally:
            self.unload_target()

    def close(self):
        self.store.close()
