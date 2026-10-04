from __future__ import annotations

import time


class Reporter:
    def __init__(self, target_id, total_tasks, enabled=True, asr_denominator=28, show_text=True, text_limit=1500):
        self.target_id = str(target_id)
        self.total_tasks = int(total_tasks)
        self.enabled = bool(enabled)
        self.asr_denominator = int(asr_denominator)
        self.show_text = bool(show_text)
        self.text_limit = max(100, int(text_limit))
        self.started = time.perf_counter()
        self.target_queries = 0
        self.successful_tasks = 0

    @staticmethod
    def _elapsed(seconds):
        seconds = max(0, int(seconds))
        return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"

    def line(self, message):
        if self.enabled:
            print(message, flush=True)

    def text_block(self, label, text, detail=""):
        if not (self.enabled and self.show_text):
            return
        value = str(text or "")
        clipped = value[: self.text_limit]
        if len(value) > self.text_limit:
            clipped += f"\n[TRUNCATED after {self.text_limit} characters]"
        suffix = f" | {detail}" if detail else ""
        print(f"[{label} START]{suffix}", flush=True)
        print(clipped, flush=True)
        print(f"[{label} END]", flush=True)

    def run_start(self, resumed_tasks=0):
        self.line(
            f"[RUN START] target={self.target_id} | tasks={self.total_tasks} | resumed_complete_tasks={resumed_tasks} "
            "| baseline=1 | adaptive_budget=15 | one_candidate_per_turn=YES"
        )

    def target_load_start(self, model):
        self.line(f"[TARGET LOAD] {self.target_id} starting | model={model}")

    def target_load_complete(self, seconds):
        self.line(f"[TARGET LOAD] {self.target_id} ready | elapsed={self._elapsed(seconds)}")

    def target_unload(self):
        self.line(f"[TARGET UNLOAD] {self.target_id} releasing model memory")

    def target_unloaded(self):
        self.line(f"[TARGET UNLOAD] {self.target_id} complete")

    def task_start(self, index, assignment_id):
        self.line(
            f"[TASK START] target={self.target_id} | task={index}/{self.total_tasks} | assignment={assignment_id} "
            f"| run_elapsed={self._elapsed(time.perf_counter() - self.started)}"
        )

    def baseline_start(self, index, assignment_id):
        self.line(f"[BASELINE] task={index}/{self.total_tasks} | assignment={assignment_id} | target_query=START")

    def stage_start(self, index, assignment_id, stage, context_decision, context_id):
        self.line(
            f"[STAGE {stage} START] task={index}/{self.total_tasks} | assignment={assignment_id} "
            f"| context={context_decision['mode']} | reason={context_decision['reason']} | context_id={context_id}"
        )

    def context_restart(self, index, assignment_id, stage, turn, reason, context_id):
        self.line(
            f"[CONTEXT RESTART] task={index}/{self.total_tasks} | assignment={assignment_id} | stage={stage} "
            f"| before_turn={turn} | reason={reason} | new_context_id={context_id}"
        )

    def provider_start(self, role, index, assignment_id, stage, turn, attempt=None):
        suffix = f" | attempt={attempt}" if attempt is not None else ""
        self.line(
            f"[CALL START] role={role} | task={index}/{self.total_tasks} | assignment={assignment_id} "
            f"| stage={stage} | turn={turn}{suffix}"
        )

    def provider_end(self, role, status, seconds, detail=""):
        suffix = f" | {detail}" if detail else ""
        self.line(f"[CALL END] role={role} | status={status} | latency={seconds:.2f}s{suffix}")

    def target_query_start(self, index, assignment_id, stage, turn, action, context_id):
        self.line(
            f"[TARGET QUERY START] task={index}/{self.total_tasks} | assignment={assignment_id} | stage={stage} "
            f"| turn={turn} | action={action} | context_id={context_id}"
        )

    def turn(self, index, assignment_id, stage, state, action, judge, reward, decision, latencies, context_id):
        chcs = judge.get("chcs") if judge.get("chcs") is not None else "UNRESOLVED"
        self.line(
            f"[TURN COMPLETE] task={index}/{self.total_tasks} | assignment={assignment_id} | stage={stage} "
            f"| stage_turn={state.turn_index}/5 | context_turn={state.context_turn_index} | context_id={context_id} "
            f"| action={action} | CHCS={chcs} | class={judge.get('response_class')} "
            f"| peak={state.peak_chcs} | stagnation={state.stagnation_count} "
            f"| success={'YES' if judge.get('final_success') else 'NO'} | reward={reward:+.3f} "
            f"| target_queries={self.target_queries} | planner={latencies.get('planner', 0):.2f}s "
            f"gate={latencies.get('gate', 0):.2f}s target={latencies.get('target', 0):.2f}s "
            f"judge={latencies.get('judge', 0):.2f}s | allowed={','.join(decision.get('allowed_actions', []))}"
        )

    def stage_complete(self, index, assignment_id, stage, success, turns, reward, reason):
        self.line(
            f"[STAGE {stage} COMPLETE] task={index}/{self.total_tasks} | assignment={assignment_id} "
            f"| success={'YES' if success else 'NO'} | target_turns={turns} | reward={reward:+.3f} | reason={reason}"
        )

    def technical_failure(self, index, assignment_id, stage, turn, reason):
        self.line(
            f"[TECHNICAL FAILURE] task={index}/{self.total_tasks} | assignment={assignment_id} "
            f"| stage={stage} | turn={turn} | reason={reason} | counted_as_ordinary_failure=NO"
        )

    def task_complete(self, index, assignment_id, success, phase, technical_incomplete, task_queries):
        if success:
            self.successful_tasks += 1
        self.line(
            f"[TASK COMPLETE] target={self.target_id} | task={index}/{self.total_tasks} | assignment={assignment_id} "
            f"| success={'YES' if success else 'NO'} | success_phase={phase} "
            f"| technical_incomplete={'YES' if technical_incomplete else 'NO'} | task_queries={task_queries} "
            f"| cumulative_successes={self.successful_tasks}/{self.asr_denominator} "
            f"| cumulative_ASR={100 * self.successful_tasks / self.asr_denominator:.1f}%"
        )

    def recovery(self, attempted, recovered, remaining):
        self.line(
            f"[JUDGE RECOVERY] target={self.target_id} | attempted={attempted} | recovered={recovered} | remaining={remaining}"
        )

    def run_complete(self, summary):
        result = summary.get("bounded_first_success", {})
        self.line(
            f"[RUN COMPLETE] target={self.target_id} | successes={result.get('successes', 0)}/28 "
            f"| ASR={100 * float(result.get('asr') or 0):.1f}% | target_queries={self.target_queries} "
            f"| elapsed={self._elapsed(time.perf_counter() - self.started)}"
        )
