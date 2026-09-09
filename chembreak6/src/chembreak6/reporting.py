from __future__ import annotations

import json
import html
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .utils import utc_now


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


class LiveReporter:
    """A file-backed, notebook-safe run dashboard with no raw model content."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        total: int,
        phase: str,
        dry_run: bool,
        refresh_seconds: float = 15.0,
        heartbeat_seconds: float = 60.0,
        baseline_completed: int = 0,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.run_dir / "live_status.json"
        self.event_path = self.run_dir / "event_log.jsonl"
        self.total = int(total)
        self.phase = phase
        self.dry_run = bool(dry_run)
        self.refresh_seconds = max(1.0, float(refresh_seconds))
        self.heartbeat_seconds = max(5.0, float(heartbeat_seconds))
        self.baseline_completed = int(baseline_completed)
        self.started = time.monotonic()
        self.last_published = 0.0
        self.display_handle: Any | None = None
        self.last_text = ""
        self._lock = threading.Lock()
        self.state: dict[str, Any] = {
            "namespace": "CB6",
            "mode": "MOCK VALIDATION" if dry_run else "LIVE JAILBREAK EVALUATION",
            "phase": phase,
            "stage": "starting",
            "total_episodes": self.total,
            "current_target": None,
            "current_condition": None,
            "current_assignment": None,
            "current_turn": None,
            "last_episode": None,
            "updated_at_utc": utc_now(),
        }
        self._configure_notebook_display()

    def _configure_notebook_display(self) -> None:
        try:
            from IPython import get_ipython
            from IPython.display import HTML, display

            shell = get_ipython()
            if shell is not None and "IPKernelApp" in getattr(shell, "config", {}):
                self.display_handle = display(
                    HTML("<pre>ChemBreak6 dashboard starting...</pre>"), display_id=True
                )
        except Exception:
            self.display_handle = None

    def event(self, name: str, **fields: Any) -> None:
        record = {"time_utc": utc_now(), "event": name, **fields}
        with self.event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")

    def render(self) -> str:
        state = self.state
        completed = int(state.get("completed", 0))
        successes = int(state.get("successes", 0))
        failures = int(state.get("failures", 0))
        technical_failed = int(state.get("technical_failed", 0))
        target_unavailable = int(state.get("target_unavailable", 0))
        pending_judgment = int(state.get("pending_judgment", 0))
        ready = int(state.get("ready", 0))
        unjudged = int(state.get("unjudged_target_responses", 0))
        queries = int(state.get("queries", 0))
        refusals = int(state.get("refusals", 0))
        asr = successes / completed if completed else 0.0
        elapsed = time.monotonic() - self.started
        accounted = completed + technical_failed + target_unavailable + pending_judgment + ready
        session_completed = max(0, accounted - self.baseline_completed)
        eta = None
        if session_completed and accounted < self.total:
            eta = elapsed / session_completed * (self.total - accounted)
        lines = [
            f"CHEMBREAK6 | {state['mode']} | phase={self.phase} | stage={state.get('stage')}",
            f"Evaluated: {completed}/{self.total} | coverage={completed / self.total if self.total else 0:.1%} | "
            f"success={successes} | ASR among evaluated={asr:.1%}",
            f"Technical status: pending judgment={pending_judgment} | ready to resume={ready} | "
            f"failed={technical_failed} | target unavailable={target_unavailable}",
            f"Audit: saved awaiting judgment={unjudged} | historical failure events={failures}",
            f"Turn totals: refusals={refusals} | target queries={queries}",
            f"Current: target={state.get('current_target') or '-'} | "
            f"condition={state.get('current_condition') or '-'} | "
            f"task={state.get('current_assignment') or '-'} | turn={state.get('current_turn') or '-'}",
            f"Timing: elapsed={_duration(elapsed)} | ETA={_duration(eta)} | "
            f"checkpoint={state.get('checkpoint') or '-'}",
        ]
        groups = state.get("by_target_condition") or []
        if groups:
            lines.extend(["", "Target / condition results:", "target    condition             done  success  ASR    mean queries"])
            for row in groups:
                group_asr = float(row.get("attack_success_rate", 0.0))
                lines.append(
                    f"{str(row.get('target_id', '-')):<9} "
                    f"{str(row.get('condition', '-')):<21} "
                    f"{int(row.get('completed', 0)):>4}  {int(row.get('successes', 0)):>7}  "
                    f"{group_asr:>5.1%}  {float(row.get('mean_queries', 0.0)):>12.2f}"
                )
        last = state.get("last_episode")
        if last:
            lines.extend(
                [
                    "",
                    "Last episode:",
                    f"{last.get('assignment_id')} | {last.get('target_id')} | {last.get('condition')} | "
                    f"queries={last.get('queries_used')} | class={last.get('final_response_class')} | "
                    f"chemistry={last.get('chemistry_validation')} | "
                    f"JAILBREAK SUCCESS={last.get('success_label')} | terminal={last.get('terminal_reason')}",
                ]
            )
        if self.dry_run:
            lines.extend(["", "MOCK MODE: these are pipeline checks, not real jailbreak results."])
        return "\n".join(lines)

    def update(
        self,
        snapshot: dict[str, Any] | None = None,
        *,
        stage: str | None = None,
        current_target: str | None = None,
        current_condition: str | None = None,
        current_assignment: str | None = None,
        current_turn: int | None = None,
        last_episode: dict[str, Any] | None = None,
        checkpoint: str | None = None,
        force: bool = False,
    ) -> None:
        with self._lock:
            if snapshot:
                self.state.update(snapshot)
            for key, value in {
                "stage": stage,
                "current_target": current_target,
                "current_condition": current_condition,
                "current_assignment": current_assignment,
                "current_turn": current_turn,
                "last_episode": last_episode,
                "checkpoint": checkpoint,
            }.items():
                if value is not None:
                    self.state[key] = value
            self.state["updated_at_utc"] = utc_now()
            self.state["elapsed_seconds"] = round(time.monotonic() - self.started, 3)
            _atomic_json(self.status_path, self.state)
            now = time.monotonic()
            if not force and now - self.last_published < self.refresh_seconds:
                return
            self.last_published = now
            rendered = self.render()
            self.last_text = rendered
            if self.display_handle is not None:
                from IPython.display import HTML

                self.display_handle.update(HTML(f"<pre>{html.escape(rendered)}</pre>"))
            else:
                print(rendered, file=sys.stdout, flush=True)

    @contextmanager
    def heartbeat(self, *, stage: str, target_id: str) -> Iterator[None]:
        stopped = threading.Event()

        def pulse() -> None:
            while not stopped.wait(self.heartbeat_seconds):
                self.event("heartbeat", stage=stage, target_id=target_id)
                self.update(stage=stage, current_target=target_id, force=True)

        self.update(stage=stage, current_target=target_id, force=True)
        worker = threading.Thread(target=pulse, daemon=True)
        worker.start()
        try:
            yield
        finally:
            stopped.set()
            worker.join(timeout=1.0)
