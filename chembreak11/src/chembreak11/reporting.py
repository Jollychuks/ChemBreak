from __future__ import annotations

import html
import json
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .utils import utc_now


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "unknown"
    value = int(seconds)
    hours, remainder = divmod(value, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {seconds:02d}s"


class LiveReporter:
    def __init__(
        self, run_dir: str | Path, *, total: int, phase: str, condition: str,
        target_scope: str,
        dry_run: bool, refresh_seconds: float = 15, heartbeat_seconds: float = 60,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.run_dir / "live_status.json"
        self.event_path = self.run_dir / "event_log.jsonl"
        self.total = int(total)
        self.phase = phase
        self.condition = condition
        self.target_scope = target_scope
        self.dry_run = bool(dry_run)
        self.refresh_seconds = max(1.0, float(refresh_seconds))
        self.heartbeat_seconds = max(5.0, float(heartbeat_seconds))
        self.started = time.monotonic()
        self.last_published = 0.0
        self.display_handle: Any | None = None
        self._lock = threading.Lock()
        self.state: dict[str, Any] = {
            "namespace": "CB11", "mode": "MOCK VALIDATION" if dry_run else "LIVE EVALUATION",
            "phase": phase, "condition": condition, "target_scope": target_scope,
            "stage": "starting", "current_target": None, "current_assignment": None,
            "current_turn": None, "total_episodes": total, "updated_at_utc": utc_now(),
        }
        self._configure_display()

    def _configure_display(self) -> None:
        try:
            from IPython import get_ipython
            from IPython.display import HTML, display

            if get_ipython() is not None:
                self.display_handle = display(HTML("<pre>ChemBreak11 starting...</pre>"), display_id=True)
        except Exception:  # noqa: BLE001
            self.display_handle = None

    def event(self, name: str, **fields: Any) -> None:
        with self.event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time_utc": utc_now(), "event": name, **fields}, sort_keys=True) + "\n")

    def render(self) -> str:
        state = self.state
        completed = int(state.get("completed", 0))
        successes = int(state.get("successes", 0))
        bootstrap = int(state.get("bootstrap_successes", 0))
        adaptive = int(state.get("adaptive_successes", 0))
        queries = int(state.get("queries", 0))
        elapsed = time.monotonic() - self.started
        lines = [
            f"CHEMBREAK11 | {state['mode']} | phase={self.phase}",
            f"Scope: {self.condition} / {self.target_scope} | stage={state.get('stage')}",
            f"Completed: {completed}/{self.total} | coverage={completed / self.total if self.total else 0:.1%}",
            f"Verified success: {successes} | overall ASR={successes / completed if completed else 0:.1%}",
            f"Bootstrap success at turn 1: {bootstrap} | post-feedback adaptive success: {adaptive}",
            (
                f"Target queries: {queries} | fully verified turns={int(state.get('fully_verified_turns', 0))} | "
                f"screened turns={int(state.get('screened_turns', 0))}"
            ),
            (
                f"Pending verification={int(state.get('pending_verification', 0))} | "
                f"ready to resume={int(state.get('ready', 0))} | historical failures={int(state.get('failures', 0))}"
            ),
            (
                f"Current: target={state.get('current_target') or '-'} | "
                f"task={state.get('current_assignment') or '-'} | turn={state.get('current_turn') or '-'}"
            ),
            f"Elapsed: {_duration(elapsed)} | checkpoint={state.get('checkpoint') or '-'}",
        ]
        last = state.get("last_episode")
        if last:
            lines.extend([
                "", "Last episode:",
                (
                    f"{last.get('assignment_id')} | {last.get('target_id')} | queries={last.get('queries_used')} | "
                    f"verified={last.get('verification_status')} | success={last.get('success_label')} | "
                    f"bootstrap={last.get('bootstrap_success')} | adaptive={last.get('adaptive_success')} | "
                    f"terminal={last.get('terminal_reason')}"
                ),
            ])
        if self.dry_run:
            lines.extend(["", "MOCK MODE: these validate the pipeline and are not real safety results."])
        return "\n".join(lines)

    def update(self, snapshot: dict[str, Any] | None = None, *, force: bool = False, **fields: Any) -> None:
        with self._lock:
            if snapshot:
                self.state.update(snapshot)
            self.state.update({key: value for key, value in fields.items() if value is not None})
            self.state["updated_at_utc"] = utc_now()
            self.state["elapsed_seconds"] = round(time.monotonic() - self.started, 3)
            _atomic_json(self.status_path, self.state)
            now = time.monotonic()
            if not force and now - self.last_published < self.refresh_seconds:
                return
            self.last_published = now
            rendered = self.render()
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
            worker.join(timeout=1)
