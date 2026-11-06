from __future__ import annotations

import json
import time


class Reporter:
    def __init__(self, target_id, total_tasks, enabled=True):
        self.target_id = str(target_id)
        self.total_tasks = int(total_tasks)
        self.enabled = bool(enabled)
        self.started = time.perf_counter()
        self.ofs5_tasks = 0
        self.target_queries = 0
        self.current_task_ofs5 = False

    def line(self, message=""):
        if self.enabled:
            print(message, flush=True)

    def run_start(self, judge_experiment, resumed_tasks=0):
        self.line(f"CHEMBREAK | {self.target_id} | JUDGE {str(judge_experiment).upper()} | TASKS {self.total_tasks}")
        if resumed_tasks:
            self.line(f"RESUMED TASKS: {int(resumed_tasks)}")

    def target_load_start(self, model):
        self.line(f"TARGET LOAD | {self.target_id} | {model}")

    def target_load_complete(self, seconds):
        self.line(f"TARGET READY | {self.target_id} | {seconds:.1f}s")

    def target_unloaded(self):
        return None

    def task_start(self, index, assignment_id):
        self.current_task_ofs5 = False

    def baseline_start(self, index, assignment_id):
        self.line()
        self.line(f"CHEMBREAK | {self.target_id}")
        self.line(f"TASK {index:02d}/{self.total_tasks:02d} | DIRECT BASELINE | {assignment_id}")

    @staticmethod
    def _score(metric):
        if metric.get("score") is None:
            return "UNRESOLVED"
        suffix = "" if metric.get("decision_status") == "valid" else f" [{metric.get('decision_status', 'UNRESOLVED').upper()}]"
        return f"{metric['score']}/5{suffix}"

    def result(self, index, stage, turn, judgment):
        if judgment.get("terminal_ofs5"):
            self.current_task_ofs5 = True
        displayed_ofs5 = self.ofs5_tasks + (1 if self.current_task_ofs5 else 0)
        self.line()
        self.line(f"CHEMBREAK | {self.target_id}")
        self.line(f"TASK {index:02d}/{self.total_tasks:02d} | STAGE {stage}/3 | TURN {turn}/6")
        self.line()
        self.line(f"RHS: {self._score(judgment['rhs'])}")
        self.line(f"OFS: {self._score(judgment['ofs'])}")
        self.line()
        self.line(f"OFS_5_TASKS: {displayed_ofs5}/{self.total_tasks}")

    def baseline_result(self, index, judgment):
        if judgment.get("terminal_ofs5"):
            self.current_task_ofs5 = True
        displayed_ofs5 = self.ofs5_tasks + (1 if self.current_task_ofs5 else 0)
        self.line(f"RHS: {self._score(judgment['rhs'])}")
        self.line(f"OFS: {self._score(judgment['ofs'])}")
        self.line(f"OFS_5_TASKS: {displayed_ofs5}/{self.total_tasks}")

    def stage_start(self, index, assignment_id, stage, context_decision, context_id, resumed_turn=0):
        if resumed_turn:
            self.line(
                f"RESUME | TASK {index:02d}/{self.total_tasks:02d} | "
                f"STAGE {stage}/3 | AFTER TURN {resumed_turn}/6"
            )

    def context_restart(self, index, assignment_id, stage, turn, reason, context_id):
        return None

    def detailed_error(self, *, index, assignment_id, stage, turn, metric, error):
        self.line()
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

    def technical_failure(self, index, assignment_id, stage, turn, reason):
        self.line(
            f"TECHNICAL ERROR | TASK {index:02d}/{self.total_tasks:02d} | {assignment_id} | "
            f"STAGE {stage}/3 | TURN {turn}/6 | {reason}"
        )

    def task_complete(self, index, assignment_id, terminal_ofs5, technical_incomplete, task_queries):
        if terminal_ofs5:
            self.ofs5_tasks += 1
        self.current_task_ofs5 = False
        self.line(
            f"TASK COMPLETE | {index:02d}/{self.total_tasks:02d} | {assignment_id} | "
            f"OFS_5={'YES' if terminal_ofs5 else 'NO'}"
        )
        self.line(f"OFS_5_TASKS: {self.ofs5_tasks}/{self.total_tasks}")

    def run_complete(self, summary):
        self.line()
        self.line(f"CHEMBREAK COMPLETE | {self.target_id}")
        self.line(f"OFS_5_TASKS: {summary.get('ofs_5_tasks', 0)}/{self.total_tasks}")
        self.line(f"UNRESOLVED RHS: {summary.get('unresolved_rhs', 0)}")
        self.line(f"UNRESOLVED OFS: {summary.get('unresolved_ofs', 0)}")
        self.line(f"TARGET QUERIES: {self.target_queries}")
