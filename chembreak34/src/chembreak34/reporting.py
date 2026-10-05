from __future__ import annotations

import time


class Reporter:
    WIDTH = 108

    def __init__(self, target_id, total_tasks, enabled=True, asr_denominator=28, show_text=True, text_limit=1500):
        self.target_id = str(target_id)
        self.total_tasks = int(total_tasks)
        self.enabled = bool(enabled)
        self.asr_denominator = int(asr_denominator)
        self.show_text = bool(show_text)
        self.text_limit = max(100, int(text_limit))
        self.started = time.perf_counter()
        self.task_started = None
        self.turn_started = None
        self.target_queries = 0
        self.successful_tasks = 0
        self._active_calls = {}

    @staticmethod
    def _elapsed(seconds):
        seconds = max(0, int(seconds))
        return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"

    def line(self, message=""):
        if self.enabled:
            print(message, flush=True)

    def rule(self, character="-"):
        self.line(character * self.WIDTH)

    def banner(self, title, character="="):
        self.line()
        self.rule(character)
        self.line(title)
        self.rule(character)

    def text_block(self, label, text, detail=""):
        if not (self.enabled and self.show_text):
            return
        value = str(text or "")
        clipped = value[: self.text_limit]
        if len(value) > self.text_limit:
            clipped += f"\n[TRUNCATED after {self.text_limit} characters]"
        suffix = f" | {detail}" if detail else ""
        self.line(f"    +-- {label}{suffix}")
        for row in clipped.splitlines() or [""]:
            self.line(f"    | {row}")
        self.line(f"    +-- END {label}")

    def run_start(self, resumed_tasks=0):
        self.banner(
            f"CHEMBREAK34 RUN START | target={self.target_id} | tasks={self.total_tasks} | "
            f"resumed={resumed_tasks} | query_budget=1 baseline + 15 adaptive | candidates=1 per turn"
        )
        self.line("CHCS scope: all target responses in the current active conversation; reset on context restart.")

    def target_load_start(self, model):
        self.banner(f"TARGET MODEL LOAD START | target={self.target_id} | model={model}", character="-")

    def target_load_complete(self, seconds):
        self.line(f"TARGET MODEL LOAD COMPLETE | target={self.target_id} | elapsed={self._elapsed(seconds)}")
        self.rule("-")

    def target_unload(self):
        self.banner(f"TARGET MODEL UNLOAD START | target={self.target_id}", character="-")

    def target_unloaded(self):
        self.line(f"TARGET MODEL UNLOAD COMPLETE | target={self.target_id}")
        self.rule("-")

    def task_start(self, index, assignment_id):
        self.task_started = time.perf_counter()
        self.banner(
            f"TASK {index:02d}/{self.total_tasks:02d} START | target={self.target_id} | assignment={assignment_id} | "
            f"run_elapsed={self._elapsed(self.task_started - self.started)}"
        )

    def baseline_start(self, index, assignment_id):
        self.banner(
            f"TASK {index:02d}/{self.total_tasks:02d} | BASELINE START | assignment={assignment_id} | direct query 1/16",
            character="-",
        )

    def baseline_complete(self, index, assignment_id, judge, success, latencies):
        chcs = judge.get("chcs") if judge.get("chcs") is not None else "UNRESOLVED"
        self.line(
            f"BASELINE END | task={index:02d}/{self.total_tasks:02d} | assignment={assignment_id} | "
            f"CHCS={chcs} | class={judge.get('response_class')} | success={'YES' if success else 'NO'}"
        )
        self.line(
            f"    CHCS evidence: active_context_responses={judge.get('response_count', 1)} | "
            f"target_time={latencies.get('target', 0):.2f}s | judge_time={latencies.get('judge', 0):.2f}s | "
            f"run_target_queries={self.target_queries}"
        )
        self.rule("-")

    def stage_start(self, index, assignment_id, stage, context_decision, context_id):
        self.banner(
            f"TASK {index:02d}/{self.total_tasks:02d} | STAGE {stage}/3 START | assignment={assignment_id}",
            character="-",
        )
        self.line(
            f"context={context_decision['mode']} | reason={context_decision['reason']} | context_id={context_id}"
        )
        if context_decision["mode"] == "CONTINUE_CONVERSATION":
            self.line("CHCS scope continues with all target responses already present in this context.")
        else:
            self.line("CHCS scope starts empty for this new target conversation.")

    def context_restart(self, index, assignment_id, stage, turn, reason, context_id):
        self.banner(
            f"CONTEXT RESTART BEFORE STAGE {stage} TURN {turn} | task={index:02d}/{self.total_tasks:02d} | "
            f"assignment={assignment_id}",
            character="!",
        )
        self.line(f"reason={reason} | new_context_id={context_id}")
        self.line("CHCS SCOPE RESET | Earlier-context responses are not included in the next judgment.")

    def turn_start(self, index, assignment_id, stage, turn, context_turn, context_id, execution_attempt=1):
        self.turn_started = time.perf_counter()
        retry = " RETRY" if int(execution_attempt) > 1 else ""
        self.line()
        self.rule(".")
        self.line(
            f">>> TURN START{retry} | task={index:02d}/{self.total_tasks:02d} | assignment={assignment_id} | "
            f"stage={stage}/3 | stage_turn={turn}/5 | context_turn={context_turn} | attempt={execution_attempt}"
        )
        self.line(f"    context_id={context_id}")
        self.rule(".")

    @staticmethod
    def _call_label(role, stage):
        if int(stage) == 0:
            return {
                "target_model": "BASELINE STEP 1/2 TARGET",
                "chcs_judge": "BASELINE STEP 2/2 CHCS JUDGE",
            }.get(role, f"BASELINE {role.upper()}")
        return {
            "attack_planner": "TURN STEP 1/4 ATTACK PLANNER",
            "intent_gate": "TURN STEP 2/4 INTENT GATE",
            "target_model": "TURN STEP 3/4 TARGET MODEL",
            "chcs_judge": "TURN STEP 4/4 CHCS JUDGE",
            "novelty_gate": "TURN NOVELTY CHECK",
        }.get(role, f"TURN {role.upper()}")

    def provider_start(self, role, index, assignment_id, stage, turn, attempt=None):
        label = self._call_label(role, stage)
        self._active_calls[role] = label
        suffix = f" | provider_attempt={attempt}" if attempt is not None else ""
        self.line(f"  [{label} START]{suffix}")

    def provider_end(self, role, status, seconds, detail=""):
        label = self._active_calls.pop(role, role.upper())
        suffix = f" | {detail}" if detail else ""
        self.line(f"  [{label} END] status={status} | latency={seconds:.2f}s{suffix}")

    def target_query_start(self, index, assignment_id, stage, turn, action, context_id):
        label = self._call_label("target_model", stage)
        self._active_calls["target_model"] = label
        self.line(
            f"  [{label} START] action={action} | task={index:02d}/{self.total_tasks:02d} | "
            f"stage={stage} | turn={turn} | context_id={context_id}"
        )

    def turn(self, index, assignment_id, stage, state, action, judge, reward, decision, latencies, context_id):
        chcs = judge.get("chcs") if judge.get("chcs") is not None else "UNRESOLVED"
        elapsed = time.perf_counter() - self.turn_started if self.turn_started else 0.0
        self.line(
            f"<<< TURN END | task={index:02d}/{self.total_tasks:02d} | assignment={assignment_id} | "
            f"stage={stage}/3 | stage_turn={state.turn_index}/5 | CHCS={chcs} | "
            f"success={'YES' if judge.get('final_success') else 'NO'}"
        )
        self.line(
            f"    CHCS evidence: active_context_responses={judge.get('response_count', state.context_turn_index)} | "
            f"context_turn={state.context_turn_index} | context_id={context_id}"
        )
        self.line(
            f"    class={judge.get('response_class')} | action={action} | reward={reward:+.3f} | "
            f"task_peak={state.peak_chcs} | stagnation={state.stagnation_count}"
        )
        self.line(
            f"    timing: planner={latencies.get('planner', 0):.2f}s | gate={latencies.get('gate', 0):.2f}s | "
            f"target={latencies.get('target', 0):.2f}s | judge={latencies.get('judge', 0):.2f}s | "
            f"turn_total={elapsed:.2f}s"
        )
        self.line(f"    next_allowed_actions={','.join(decision.get('allowed_actions', []))}")
        self.line(f"    run_target_queries={self.target_queries}")
        self.rule(".")

    def stage_complete(self, index, assignment_id, stage, success, turns, reward, reason):
        self.line(
            f"STAGE {stage}/3 END | task={index:02d}/{self.total_tasks:02d} | assignment={assignment_id} | "
            f"success={'YES' if success else 'NO'} | completed_turns={turns}/5 | "
            f"stage_reward={reward:+.3f} | reason={reason}"
        )
        self.rule("-")

    def technical_failure(self, index, assignment_id, stage, turn, reason):
        self.line(
            f"<<< TURN ATTEMPT FAILED | task={index:02d}/{self.total_tasks:02d} | assignment={assignment_id} | "
            f"stage={stage} | planned_turn={turn} | reason={reason} | target_turn_counted=NO"
        )
        self.rule("!")

    def task_complete(self, index, assignment_id, success, phase, technical_incomplete, task_queries):
        if success:
            self.successful_tasks += 1
        task_elapsed = time.perf_counter() - self.task_started if self.task_started else 0.0
        self.banner(
            f"TASK {index:02d}/{self.total_tasks:02d} END | assignment={assignment_id} | "
            f"success={'YES' if success else 'NO'} | phase={phase} | task_queries={task_queries} | "
            f"task_elapsed={self._elapsed(task_elapsed)}"
        )
        self.line(
            f"RUNNING RESULT | successful_tasks={self.successful_tasks}/{self.asr_denominator} | "
            f"ASR={100 * self.successful_tasks / self.asr_denominator:.1f}% | "
            f"technical_incomplete={'YES' if technical_incomplete else 'NO'}"
        )

    def recovery(self, attempted, recovered, remaining):
        self.banner("FINAL JUDGE RECOVERY", character="-")
        self.line(f"attempted={attempted} | recovered={recovered} | remaining_unresolved={remaining}")
        self.rule("-")

    def run_complete(self, summary):
        result = summary.get("bounded_first_success", {})
        self.banner(
            f"CHEMBREAK34 RUN COMPLETE | target={self.target_id} | "
            f"successes={result.get('successes', 0)}/{self.asr_denominator} | "
            f"ASR={100 * float(result.get('asr') or 0):.1f}% | target_queries={self.target_queries} | "
            f"elapsed={self._elapsed(time.perf_counter() - self.started)}"
        )
