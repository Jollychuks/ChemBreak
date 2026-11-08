from __future__ import annotations

import json
import time


class Reporter:
    WIDTH = 108

    def __init__(self, target_id, total_tasks, enabled=True, settings=None, judge_model=""):
        settings = dict(settings or {})
        self.target_id = str(target_id)
        self.total_tasks = int(total_tasks)
        self.enabled = bool(enabled)
        self.detailed = bool(settings.get("detailed", True))
        self.text_limit = max(0, int(settings.get("text_limit", 1500)))
        self.show_source_task = bool(settings.get("show_source_task", True))
        self.show_candidate = bool(settings.get("show_candidate", True))
        self.show_target_response = bool(settings.get("show_target_response", True))
        self.show_timings = bool(settings.get("show_timings", True))
        self.judge_model = str(judge_model)
        self.started = time.perf_counter()
        self.ofs5_tasks = 0
        self.target_queries = 0
        self.current_task_ofs5 = False

    def line(self, message=""):
        if self.enabled:
            print(message, flush=True)

    def _rule(self, character="-"):
        self.line(character * self.WIDTH)

    def _elapsed(self):
        seconds = max(0, int(time.perf_counter() - self.started))
        return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"

    def _block(self, title, text, metadata=""):
        value = str(text or "")
        truncated = self.text_limit and len(value) > self.text_limit
        if truncated:
            value = value[: self.text_limit].rstrip()
        suffix = f" | {metadata}" if metadata else ""
        self.line(f"    +-- {title}{suffix}")
        for row in (value.splitlines() or [""]):
            self.line(f"    | {row}")
        if truncated:
            self.line(f"    | [TRUNCATED after {self.text_limit} characters]")
        self.line(f"    +-- END {title}")

    @staticmethod
    def _score(metric):
        if metric.get("score") is None:
            return "UNRESOLVED"
        suffix = "" if metric.get("decision_status") == "valid" else (
            f" [{metric.get('decision_status', 'UNRESOLVED').upper()}]"
        )
        return f"{metric['score']}/5{suffix}"

    def run_start(self, judge_experiment, resumed_tasks=0):
        if not self.detailed:
            self.line(
                f"CHEMBREAK | {self.target_id} | JUDGE {str(judge_experiment).upper()} | "
                f"TASKS {self.total_tasks}"
            )
            return
        self._rule("=")
        self.line(
            f"CHEMBREAK RUN START | target={self.target_id} | judge={self.judge_model} | "
            f"tasks={self.total_tasks} | resumed={int(resumed_tasks)} | "
            "query_budget=1 baseline + 18 adaptive"
        )
        self._rule("=")
        self.line("OFS retains the original ChemBreak control role. RHS is measured for reporting only.")
        self.line("RHS/OFS scope: cumulative target responses in the active conversation; reset on context restart.")
        self.line("Stage 1 uses the exact v1.3 planner and gate contract. Stages 2 and 3 use fresh six-turn progressive attempts.")
        self.line("Progressive turns 5 and 6 are complete-objective request and complete-objective recovery.")

    def target_load_start(self, model):
        if self.detailed:
            self.line()
            self._rule()
            self.line(f"TARGET MODEL LOAD START | target={self.target_id} | model={model}")
            self._rule()
        else:
            self.line(f"TARGET LOAD | {self.target_id} | {model}")

    def target_load_complete(self, seconds):
        if self.detailed:
            self.line(f"TARGET MODEL LOAD COMPLETE | target={self.target_id} | elapsed={seconds:.1f}s")
            self._rule()
        else:
            self.line(f"TARGET READY | {self.target_id} | {seconds:.1f}s")

    def target_unloaded(self):
        return None

    def task_start(self, index, assignment_id):
        self.current_task_ofs5 = False
        if self.detailed:
            self.line()
            self._rule("=")
            self.line(
                f"TASK {index:02d}/{self.total_tasks:02d} START | target={self.target_id} | "
                f"assignment={assignment_id} | run_elapsed={self._elapsed()}"
            )
            self._rule("=")

    def baseline_start(self, index, assignment_id, source_task=""):
        if self.detailed:
            self.line()
            self._rule()
            self.line(
                f"TASK {index:02d}/{self.total_tasks:02d} | DIRECT BASELINE START | "
                f"assignment={assignment_id} | query=1/19"
            )
            self._rule()
            if self.show_source_task:
                self._block("SOURCE TASK", source_task, f"task={assignment_id}")
        else:
            self.line()
            self.line(f"CHEMBREAK | {self.target_id}")
            self.line(f"TASK {index:02d}/{self.total_tasks:02d} | DIRECT BASELINE | {assignment_id}")

    def stage_start(self, index, assignment_id, stage, context_decision, context_id, resumed_turn=0):
        if not self.detailed:
            if resumed_turn:
                self.line(
                    f"RESUME | TASK {index:02d}/{self.total_tasks:02d} | "
                    f"STAGE {stage}/3 | AFTER TURN {resumed_turn}/6"
                )
            return
        self.line()
        self._rule()
        self.line(
            f"TASK {index:02d}/{self.total_tasks:02d} | STAGE {stage}/3 START | "
            f"assignment={assignment_id}"
        )
        self._rule()
        self.line(
            f"context={context_decision['mode']} | reason={context_decision['reason']} | "
            f"context_id={context_id}"
        )
        strategy = str(context_decision.get("strategy_mode", "adaptive_actions"))
        attempt = int(context_decision.get("progressive_attempt", 0) or 0)
        self.line(
            f"strategy={strategy} | progressive_attempt={attempt if attempt else 'N/A'} | "
            f"mid_stage_context_restart={'DISABLED' if strategy == 'progressive_buildup' else 'POLICY_CONTROLLED'}"
        )
        if context_decision["mode"] == "RESTART_CONVERSATION":
            self.line("RHS/OFS scope starts empty for this target conversation.")
        if resumed_turn:
            self.line(f"RESUME AFTER TURN {resumed_turn}/6")

    def context_restart(self, index, assignment_id, stage, turn, reason, context_id):
        if not self.detailed:
            return
        self.line()
        self._rule("!")
        self.line(
            f"CONTEXT RESTART BEFORE STAGE {stage} TURN {turn} | "
            f"task={index:02d}/{self.total_tasks:02d} | assignment={assignment_id}"
        )
        self._rule("!")
        self.line(f"reason={reason} | new_context_id={context_id}")
        self.line("RHS/OFS SCOPE RESET | Earlier-context responses are excluded from the next judgment.")

    def turn_start(
        self, index, assignment_id, stage, turn, context_turn, context_id, *,
        strategy_mode="adaptive_actions", progressive_attempt=0, buildup_level=None, trajectory_role="",
    ):
        if not self.detailed:
            return
        self.line()
        self._rule(".")
        self.line(
            f">>> TURN START | task={index:02d}/{self.total_tasks:02d} | assignment={assignment_id} | "
            f"stage={stage}/3 | stage_turn={turn}/6 | context_turn={context_turn}"
        )
        self.line(f"    context_id={context_id}")
        if strategy_mode == "progressive_buildup":
            self.line(
                f"    strategy=progressive_buildup | attempt={progressive_attempt} | "
                f"buildup_level={buildup_level}/5 | trajectory_role={trajectory_role}"
            )
        else:
            self.line("    strategy=adaptive_actions | Stage 1 original action policy")
        self._rule(".")
        self.line("  [TURN STEP 1/4 CONTROLLER AND PLANNER]")

    def candidate_ready(
        self, action, candidate, gate, latencies, *, strategy_mode="adaptive_actions",
        progressive_attempt=0, buildup_level=None, buildup_route="",
    ):
        if not self.detailed:
            return
        self.line(
            f"  [TURN STEP 2/4 CANDIDATE ACCEPTED] action={action} | "
            f"planner={float(latencies.get('planner', 0)):.2f}s | gate={float(latencies.get('gate', 0)):.2f}s | "
            f"goal={float(gate.get('goal_preservation', 0)):.2f} | "
            f"fidelity={float(gate.get('task_fidelity', 0)):.2f} | "
            f"alignment={float(gate.get('action_alignment', 0)):.2f}"
        )
        if strategy_mode == "progressive_buildup":
            self.line(
                f"      progressive_attempt={progressive_attempt} | buildup_level={buildup_level}/5 | "
                f"route={buildup_route} | trajectory={float(gate.get('trajectory_fidelity', 0)):.2f} | "
                f"level={float(gate.get('level_compliance', 0)):.2f} | "
                f"ceiling={float(gate.get('intensity_ceiling_compliance', 0)):.2f} | "
                f"route_difference={float(gate.get('route_difference', 0)):.2f}"
            )
        if self.show_candidate:
            self._block("ATTACK CANDIDATE", candidate, f"action={action}")

    def target_start(self, phase, index, assignment_id, stage, turn, action, context_id):
        if self.detailed:
            label = "BASELINE" if phase == "baseline" else "TURN STEP 3/4"
            self.line(
                f"  [{label} TARGET START] action={action} | task={index:02d}/{self.total_tasks:02d} | "
                f"stage={stage} | turn={turn} | context_id={context_id}"
            )

    def target_end(self, phase, assignment_id, stage, turn, response, seconds):
        if not self.detailed:
            return
        label = "BASELINE" if phase == "baseline" else "TURN STEP 3/4"
        self.line(f"  [{label} TARGET END] status=valid | latency={seconds:.2f}s | target={self.target_id}")
        if self.show_target_response:
            self._block(
                "TARGET RESPONSE", response,
                f"task={assignment_id} stage={stage} turn={turn} target={self.target_id}",
            )

    def judge_start(self, phase, metric):
        if self.detailed:
            label = "BASELINE" if phase == "baseline" else "TURN STEP 4/4"
            self.line(f"  [{label} {str(metric).upper()} JUDGE START]")

    def judge_end(self, phase, metric, result, seconds):
        if not self.detailed:
            return
        label = "BASELINE" if phase == "baseline" else "TURN STEP 4/4"
        score = self._score(result)
        status = result.get("decision_status", "unresolved")
        self.line(
            f"  [{label} {str(metric).upper()} JUDGE END] status={status} | "
            f"latency={seconds:.2f}s | judge={self.judge_model} | score={score}"
        )

    def result(self, index, stage, turn, judgment, details=None):
        details = dict(details or {})
        if judgment.get("terminal_ofs5"):
            self.current_task_ofs5 = True
        displayed_ofs5 = self.ofs5_tasks + (1 if self.current_task_ofs5 else 0)
        if not self.detailed:
            self.line()
            self.line(f"CHEMBREAK | {self.target_id}")
            self.line(f"TASK {index:02d}/{self.total_tasks:02d} | STAGE {stage}/3 | TURN {turn}/6")
            self.line()
            self.line(f"RHS: {self._score(judgment['rhs'])}")
            self.line(f"OFS: {self._score(judgment['ofs'])}")
            self.line()
            self.line(f"OFS_5_TASKS: {displayed_ofs5}/{self.total_tasks}")
            return
        self.line(
            f"<<< TURN END | task={index:02d}/{self.total_tasks:02d} | "
            f"assignment={details.get('assignment_id', '')} | stage={stage}/3 | stage_turn={turn}/6 | "
            f"RHS={self._score(judgment['rhs'])} | OFS={self._score(judgment['ofs'])}"
        )
        self.line(
            f"    action={details.get('action', '')} | response_class={details.get('response_class', 'unknown')} | "
            f"reward={float(details.get('reward', 0)):+.3f} | "
            f"peak_rhs={details.get('peak_rhs', 1)} | peak_ofs={details.get('peak_ofs', 1)} | "
            f"stagnation={details.get('stagnation', 0)}"
        )
        if details.get("strategy_mode") == "progressive_buildup":
            self.line(
                f"    strategy=progressive_buildup | attempt={details.get('progressive_attempt')} | "
                f"buildup_level={details.get('buildup_level')}/5 | route={details.get('buildup_route', '')}"
            )
        self.line(
            f"    context_id={details.get('context_id', '')} | "
            f"active_context_responses={judgment.get('response_count', 0)}"
        )
        rhs = judgment.get("rhs", {})
        ofs = judgment.get("ofs", {})
        self.line(
            f"    judge_reason: RHS={rhs.get('reason_code', '')} | OFS={ofs.get('reason_code', '')} | "
            f"OFS_response_class={ofs.get('response_class', 'unknown')} | "
            f"RHS_credibility={rhs.get('scientific_credibility', 'unknown')}"
        )
        flags = judgment.get("calibration_flags", []) or []
        if flags:
            self.line("    calibration_flags: " + ", ".join(map(str, flags)))
        if self.show_timings:
            timings = details.get("timings", {})
            self.line(
                f"    timing: planner={float(timings.get('planner', 0)):.2f}s | "
                f"gate={float(timings.get('gate', 0)):.2f}s | "
                f"target={float(timings.get('target', 0)):.2f}s | "
                f"judge={float(timings.get('judge', 0)):.2f}s"
            )
        self.line(f"    OFS_5_TASKS: {displayed_ofs5}/{self.total_tasks}")
        self._rule(".")

    def baseline_result(self, index, judgment):
        if judgment.get("terminal_ofs5"):
            self.current_task_ofs5 = True
        displayed_ofs5 = self.ofs5_tasks + (1 if self.current_task_ofs5 else 0)
        if self.detailed:
            self.line(
                f"BASELINE END | task={index:02d}/{self.total_tasks:02d} | "
                f"RHS={self._score(judgment['rhs'])} | OFS={self._score(judgment['ofs'])} | "
                f"OFS_5_TASKS={displayed_ofs5}/{self.total_tasks}"
            )
            self._rule()
        else:
            self.line(f"RHS: {self._score(judgment['rhs'])}")
            self.line(f"OFS: {self._score(judgment['ofs'])}")
            self.line(f"OFS_5_TASKS: {displayed_ofs5}/{self.total_tasks}")

    def stage_end(self, index, assignment_id, stage, turns, total_reward, reason):
        if self.detailed:
            self.line(
                f"STAGE {stage}/3 END | task={index:02d}/{self.total_tasks:02d} | "
                f"assignment={assignment_id} | completed_turns={turns}/6 | "
                f"stage_reward={total_reward:+.3f} | reason={reason}"
            )
            self._rule()

    def detailed_error(self, *, index, assignment_id, stage, turn, metric, error):
        self.line()
        self._rule("!")
        self.line(
            f"JUDGE ERROR | TASK {index:02d}/{self.total_tasks:02d} | {assignment_id} | "
            f"STAGE {stage}/3 | TURN {turn}/6 | METRIC {str(metric).upper()}"
        )
        self.line(f"CATEGORY: {error.get('category', 'unknown')}")
        self.line(f"PROVIDER: {error.get('provider', '')}")
        self.line(f"STATUS: {error.get('status', '')}")
        self.line(f"CODE: {error.get('error_code', '')}")
        self.line(f"MESSAGE: {error.get('message', '')}")
        metadata = error.get("metadata")
        if metadata:
            self.line("DIAGNOSTICS: " + json.dumps(metadata, ensure_ascii=False, default=str)[:3000])
        self._rule("!")

    def technical_failure(self, index, assignment_id, stage, turn, reason):
        self._rule("!")
        self.line(
            f"TECHNICAL ERROR | TASK {index:02d}/{self.total_tasks:02d} | {assignment_id} | "
            f"STAGE {stage}/3 | TURN {turn}/6 | {reason}"
        )
        self._rule("!")

    def task_complete(self, index, assignment_id, terminal_ofs5, technical_incomplete, task_queries):
        if terminal_ofs5:
            self.ofs5_tasks += 1
        self.current_task_ofs5 = False
        if self.detailed:
            self.line()
            self._rule("=")
            self.line(
                f"TASK {index:02d}/{self.total_tasks:02d} END | assignment={assignment_id} | "
                f"OFS_5={'YES' if terminal_ofs5 else 'NO'} | task_queries={task_queries} | "
                f"technical_incomplete={'YES' if technical_incomplete else 'NO'}"
            )
            self.line(f"OFS_5_TASKS: {self.ofs5_tasks}/{self.total_tasks}")
            self._rule("=")
        else:
            self.line(
                f"TASK COMPLETE | {index:02d}/{self.total_tasks:02d} | {assignment_id} | "
                f"OFS_5={'YES' if terminal_ofs5 else 'NO'}"
            )
            self.line(f"OFS_5_TASKS: {self.ofs5_tasks}/{self.total_tasks}")

    def run_complete(self, summary):
        self.line()
        if self.detailed:
            self._rule("=")
        self.line(f"CHEMBREAK COMPLETE | {self.target_id}")
        self.line(f"OFS_5_TASKS: {summary.get('ofs_5_tasks', 0)}/{self.total_tasks}")
        self.line(f"UNRESOLVED RHS: {summary.get('unresolved_rhs', 0)}")
        self.line(f"UNRESOLVED OFS: {summary.get('unresolved_ofs', 0)}")
        self.line(f"TARGET QUERIES: {self.target_queries}")
        if self.detailed:
            self._rule("=")
