from __future__ import annotations

from dataclasses import dataclass
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .benchmark import load_and_validate_task_bank, select_tasks, to_task_records, write_selection
from .checkpoint import CheckpointStore, sync_checkpoint_to_gcs
from .conditions import recover_pending_turn, run_episode
from .config import load_config, project_root, run_signature, validate_config
from .metrics import export_results
from .providers import RoleClients, StructuredOutputError
from .reporting import LiveReporter
from .schema import REGISTERED_CONDITIONS, TaskRecord
from .storage import configure_content_storage, verify_content_storage
from .targets import make_target
from .utils import require_live_gate, set_global_seed, sha256_file, stable_id, utc_now


@dataclass(slots=True)
class RunContext:
    config: dict[str, Any]
    root: Path
    bank_path: Path
    signature: str
    run_dir: Path
    selection_path: Path
    tasks: list[TaskRecord]
    store: CheckpointStore
    project_id: str | None


def _project_id(config: dict[str, Any]) -> str | None:
    if config["run"]["dry_run"]:
        return None
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT_ID")
    if not project_id:
        raise RuntimeError("Set GOOGLE_CLOUD_PROJECT to the active Google Cloud project ID.")
    return project_id


def _manifest(
    config: dict[str, Any],
    signature: str,
    bank_hash: str,
    project_id: str | None,
    storage_report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "namespace": "CB6",
        "software_version": __version__,
        "run_signature": signature,
        "task_bank_sha256": bank_hash,
        "created_at_utc": utc_now(),
        "phase": config["run"]["phase"],
        "seed": config["run"]["seed"],
        "dry_run": config["run"]["dry_run"],
        "project_id_recorded": bool(project_id),
        "storage": storage_report,
        "python": sys.version,
        "platform": platform.platform(),
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
    }


def _episode_id(
    signature: str,
    phase: str,
    target_id: str,
    condition: str,
    assignment_id: str,
) -> str:
    return (
        f"CB6-{phase}-{target_id}-{condition}-{assignment_id}-"
        f"{stable_id(signature, target_id, condition, assignment_id, length=8)}"
    )


def _initialize(config_path: str | Path) -> RunContext:
    config = load_config(config_path)
    validate_config(config)
    configure_content_storage(config)
    root = project_root(config)
    storage_report = verify_content_storage(config, root)
    require_live_gate(config)
    set_global_seed(int(config["run"]["seed"]))
    bank_setting = Path(config["run"]["task_bank_path"])
    bank_path = bank_setting.resolve() if bank_setting.is_absolute() else (root / bank_setting).resolve()
    frame = load_and_validate_task_bank(bank_path)
    selected = select_tasks(
        frame,
        int(config["experiment"]["task_count"]),
        int(config["run"]["seed"]),
    )
    bank_hash = sha256_file(bank_path)
    signature = run_signature(config, bank_hash, __version__)
    run_dir = Path(config["run"]["output_root"]).resolve() / (
        f"CB6_{config['run']['phase']}_{signature[:12]}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    selection_path = run_dir / "selected_tasks.csv"
    if selection_path.exists():
        frozen = pd.read_csv(selection_path)
        if list(frozen["assignment_id"].astype(str)) != list(selected["assignment_id"].astype(str)):
            raise RuntimeError("The existing run has a different frozen task selection.")
    else:
        write_selection(selected, selection_path)
    project_id = _project_id(config)
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_signature") != signature:
            raise RuntimeError("Existing run manifest has a different signature.")
    else:
        manifest = _manifest(config, signature, bank_hash, project_id, storage_report)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    store = CheckpointStore(
        run_dir / "state.sqlite3",
        signature,
        {"manifest": manifest, "project_root": str(root)},
    )
    return RunContext(
        config=config,
        root=root,
        bank_path=bank_path,
        signature=signature,
        run_dir=run_dir,
        selection_path=selection_path,
        tasks=to_task_records(selected),
        store=store,
        project_id=project_id,
    )


def _export(context: RunContext) -> None:
    context.store.backup(context.run_dir / "checkpoint_snapshot.sqlite3")
    export_results(
        context.store.path,
        context.selection_path,
        context.run_dir,
        context.config["experiment"]["condition_query_budgets"],
        bool(context.config["run"].get("release_raw_outputs", False)),
    )


def run_condition(config_path: str | Path, condition: str) -> Path:
    if condition not in REGISTERED_CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    context = _initialize(config_path)
    config = context.config
    if condition not in config["experiment"]["conditions"]:
        context.store.close()
        raise ValueError(f"Condition {condition} is not enabled in this configuration.")
    total = len(context.tasks) * len(config["targets"])
    completed = context.store.completed_episode_ids(condition)
    report_config = config.get("reporting", {})
    reporter = LiveReporter(
        context.run_dir,
        total=total,
        phase=f"{config['run']['phase']}:{condition}",
        dry_run=bool(config["run"]["dry_run"]),
        refresh_seconds=float(report_config.get("refresh_seconds", 15)),
        heartbeat_seconds=float(report_config.get("heartbeat_seconds", 60)),
        baseline_completed=len(completed),
    )
    reporter.event(
        "condition_run_started",
        condition=condition,
        total_episodes=total,
        resumed_completed=len(completed),
    )
    reporter.update(
        context.store.progress_snapshot(total, condition),
        stage="starting",
        current_condition=condition,
        force=True,
    )
    clients = RoleClients(config, context.project_id)
    checkpoint_every = int(config["run"].get("checkpoint_every_episodes", 1))
    processed = 0
    try:
        for target_settings in config["targets"]:
            target_id = str(target_settings["id"])
            target_episode_ids = {
                _episode_id(
                    context.signature,
                    config["run"]["phase"],
                    target_id,
                    condition,
                    task.assignment_id,
                )
                for task in context.tasks
            }
            if target_episode_ids.issubset(context.store.completed_episode_ids(condition)):
                reporter.event("target_condition_skipped_complete", target_id=target_id, condition=condition)
                continue
            target = make_target(target_settings, bool(config["run"]["dry_run"]))
            loaded = False
            try:
                reporter.event("target_load_started", target_id=target_id, condition=condition)
                with reporter.heartbeat(stage="loading_target", target_id=target_id):
                    target.load()
                loaded = True
                reporter.event("target_loaded", target_id=target_id, condition=condition)
            except Exception as exc:
                context.store.record_failure(None, f"target_load:{target_id}:{condition}", exc)
                unavailable = [
                    (
                        _episode_id(
                            context.signature,
                            config["run"]["phase"],
                            target_id,
                            condition,
                            task.assignment_id,
                        ),
                        task.assignment_id,
                        condition,
                    )
                    for task in context.tasks
                ]
                context.store.mark_target_unavailable(target_id, unavailable, exc)
                reporter.event(
                    "target_load_failed",
                    target_id=target_id,
                    condition=condition,
                    error_type=type(exc).__name__,
                )
                reporter.update(
                    context.store.progress_snapshot(total, condition),
                    stage="target_unavailable_continuing",
                    current_target=target_id,
                    current_condition=condition,
                    force=True,
                )
                target.unload()
                continue

            try:
                for task in context.tasks:
                    episode_id = _episode_id(
                        context.signature,
                        config["run"]["phase"],
                        target_id,
                        condition,
                        task.assignment_id,
                    )
                    if context.store.episode_status(episode_id) == "complete":
                        continue
                    reporter.event(
                        "episode_started",
                        episode_id=episode_id,
                        assignment_id=task.assignment_id,
                        target_id=target_id,
                        condition=condition,
                    )
                    reporter.update(
                        context.store.progress_snapshot(total, condition),
                        stage="running_episode",
                        current_target=target_id,
                        current_condition=condition,
                        current_assignment=task.assignment_id,
                        current_turn=0,
                    )

                    def on_turn(result: dict[str, Any]) -> None:
                        reporter.event(
                            "turn_completed",
                            episode_id=result["episode_id"],
                            turn_index=result["turn_index"] if "turn_index" in result else result["queries_used"],
                            success=result["success"],
                            response_class=result["final_response_class"],
                        )
                        reporter.update(
                            context.store.progress_snapshot(total, condition),
                            stage="running_episode",
                            current_target=target_id,
                            current_condition=condition,
                            current_assignment=task.assignment_id,
                            current_turn=int(result["queries_used"]),
                            last_episode=result if result.get("terminal_reason") else None,
                        )

                    try:
                        result = run_episode(
                            episode_id=episode_id,
                            condition=condition,
                            task=task,
                            target_id=target_id,
                            target=target,
                            clients=clients,
                            store=context.store,
                            config=config,
                            on_turn=on_turn,
                        )
                        reporter.event(
                            "episode_completed",
                            episode_id=episode_id,
                            success=result["success"],
                            terminal_reason=result["terminal_reason"],
                            queries_used=result["queries_used"],
                        )
                    except StructuredOutputError as exc:
                        context.store.save_api_calls(episode_id, clients.drain_call_history())
                        context.store.record_failure(episode_id, exc.stage, exc)
                        context.store.mark_pending_judgment(episode_id, exc)
                        result = {
                            "episode_id": episode_id,
                            "assignment_id": task.assignment_id,
                            "target_id": target_id,
                            "condition": condition,
                            "queries_used": context.store.episode_queries_used(episode_id),
                            "success": None,
                            "success_label": "PENDING",
                            "final_response_class": "pending_judgment",
                            "chemistry_validation": "PENDING",
                            "terminal_reason": "pending_judgment",
                        }
                        reporter.event(
                            "episode_pending_judgment",
                            episode_id=episode_id,
                            role=exc.role,
                            attempts=exc.attempts,
                        )
                    except Exception as exc:
                        context.store.save_api_calls(episode_id, clients.drain_call_history())
                        stage = str(getattr(exc, "stage", "episode"))
                        context.store.record_failure(episode_id, stage, exc)
                        context.store.fail_episode(episode_id, exc)
                        result = {
                            "episode_id": episode_id,
                            "assignment_id": task.assignment_id,
                            "target_id": target_id,
                            "condition": condition,
                            "queries_used": context.store.episode_queries_used(episode_id),
                            "success": None,
                            "success_label": "NOT_EVALUATED",
                            "final_response_class": "technical_failure",
                            "chemistry_validation": "NOT_EVALUATED",
                            "terminal_reason": "technical_failure",
                        }
                        reporter.event(
                            "episode_failed",
                            episode_id=episode_id,
                            stage=stage,
                            error_type=type(exc).__name__,
                        )
                    processed += 1
                    checkpoint_path: str | None = None
                    if processed % checkpoint_every == 0:
                        snapshot_path = context.store.backup(
                            context.run_dir / "checkpoint_snapshot.sqlite3"
                        )
                        sync_checkpoint_to_gcs(
                            snapshot_path, config["run"].get("gcs_checkpoint_uri")
                        )
                        checkpoint_path = str(snapshot_path)
                    reporter.update(
                        context.store.progress_snapshot(total, condition),
                        stage="running",
                        current_target=target_id,
                        current_condition=condition,
                        current_assignment=task.assignment_id,
                        current_turn=int(result["queries_used"]),
                        last_episode=result,
                        checkpoint=checkpoint_path,
                        force=True,
                    )
            finally:
                if loaded:
                    target.unload()
                    reporter.event("target_unloaded", target_id=target_id, condition=condition)

        _export(context)
        snapshot = context.store.progress_snapshot(total, condition)
        incomplete = bool(
            snapshot.get("pending_judgment")
            or snapshot.get("technical_failed")
            or snapshot.get("target_unavailable")
            or snapshot.get("completed", 0) != total
        )
        stage = "condition_incomplete" if incomplete else "condition_complete"
        reporter.event("condition_run_finished", condition=condition, **snapshot)
        reporter.update(snapshot, stage=stage, current_condition=condition, force=True)
    except Exception as exc:
        try:
            context.store.save_api_calls(None, clients.drain_call_history())
            _export(context)
        except Exception as export_error:
            reporter.event(
                "partial_export_failed",
                error_type=type(export_error).__name__,
                error_message=str(export_error)[:1000],
            )
        reporter.event("condition_run_failed", condition=condition, error_type=type(exc).__name__)
        reporter.update(
            context.store.progress_snapshot(total, condition),
            stage="failed",
            current_condition=condition,
            force=True,
        )
        raise
    finally:
        context.store.close()
    return context.run_dir


def recover_pending_judgments(
    config_path: str | Path,
    condition: str | None = None,
) -> dict[str, Any]:
    if condition is not None and condition not in REGISTERED_CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    context = _initialize(config_path)
    clients = RoleClients(context.config, context.project_id)
    task_map = {task.assignment_id: task for task in context.tasks}
    rows = context.store.pending_episode_rows(condition)
    recovered = 0
    still_pending = 0
    failed = 0
    try:
        for row in rows:
            episode_id = str(row["episode_id"])
            try:
                recover_pending_turn(
                    episode_id=episode_id,
                    condition=str(row["condition"]),
                    task=task_map[str(row["assignment_id"])],
                    target_id=str(row["target_id"]),
                    clients=clients,
                    store=context.store,
                    config=context.config,
                )
                recovered += 1
            except StructuredOutputError as exc:
                context.store.save_api_calls(episode_id, clients.drain_call_history())
                context.store.record_failure(episode_id, exc.stage, exc)
                context.store.mark_pending_judgment(episode_id, exc)
                still_pending += 1
            except Exception as exc:
                context.store.save_api_calls(episode_id, clients.drain_call_history())
                context.store.record_failure(
                    episode_id, str(getattr(exc, "stage", "recovery")), exc
                )
                context.store.fail_episode(episode_id, exc)
                failed += 1
        _export(context)
        remaining = len(context.store.pending_episode_rows(condition))
        return {
            "run_dir": str(context.run_dir),
            "attempted": len(rows),
            "recovered": recovered,
            "still_pending": max(still_pending, remaining),
            "failed": failed,
        }
    finally:
        context.store.close()


def finalize_run(config_path: str | Path, require_complete: bool = True) -> dict[str, Any]:
    context = _initialize(config_path)
    total = len(context.tasks) * len(context.config["targets"]) * len(
        context.config["experiment"]["conditions"]
    )
    try:
        _export(context)
        snapshot = context.store.progress_snapshot(total)
        complete = bool(
            snapshot["completed"] == total
            and snapshot["pending_judgment"] == 0
            and snapshot["technical_failed"] == 0
            and snapshot["target_unavailable"] == 0
        )
        status = {
            "run_dir": str(context.run_dir),
            "complete": complete,
            **snapshot,
        }
        (context.run_dir / "run_status.json").write_text(
            json.dumps(status, indent=2, sort_keys=True), encoding="utf-8"
        )
        if require_complete and not complete:
            raise RuntimeError(
                "ChemBreak6 is incomplete. Run the missing condition cells and recovery cell before finalizing. "
                f"Status: {status}"
            )
        return status
    finally:
        context.store.close()


def run(config_path: str | Path) -> Path:
    run_dir: Path | None = None
    config = load_config(config_path)
    validate_config(config)
    for condition in config["experiment"]["conditions"]:
        run_dir = run_condition(config_path, condition)
    assert run_dir is not None
    finalize_run(config_path, require_complete=False)
    return run_dir
