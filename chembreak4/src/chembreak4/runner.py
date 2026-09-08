from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .benchmark import load_and_validate_task_bank, select_tasks, to_task_records, write_selection
from .checkpoint import CheckpointStore, sync_checkpoint_to_gcs
from .conditions import run_episode
from .config import load_config, resolve_project_path, run_signature, validate_config
from .metrics import export_results
from .providers import RoleClients
from .reporting import LiveReporter
from .storage import configure_content_storage, verify_content_storage
from .targets import make_target
from .utils import require_live_gate, set_global_seed, sha256_file, stable_id, utc_now


def _project_id(config: dict[str, Any]) -> str | None:
    if config["run"]["dry_run"]:
        return None
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT_ID")
    if not project_id:
        raise RuntimeError("Set GOOGLE_CLOUD_PROJECT to the active Google Cloud project ID.")
    return project_id


def _manifest(
    config: dict[str, Any], signature: str, bank_hash: str,
    project_id: str | None, storage_report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "namespace": "CB4",
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


def _episode_id(signature: str, phase: str, target_id: str, condition: str, assignment_id: str) -> str:
    return (
        f"CB4-{phase}-{target_id}-{condition}-{assignment_id}-"
        f"{stable_id(signature, target_id, condition, assignment_id, length=8)}"
    )


def run(config_path: str | Path) -> Path:
    config = load_config(config_path)
    validate_config(config)
    configure_content_storage(config)
    project_root = Path(config["_config_path"]).parent.parent.resolve()
    storage_report = verify_content_storage(config, project_root)
    require_live_gate(config)
    set_global_seed(int(config["run"]["seed"]))
    bank_path = resolve_project_path(config["_config_path"], config["run"]["task_bank_path"])
    frame = load_and_validate_task_bank(bank_path)
    selected = select_tasks(frame, int(config["experiment"]["task_count"]), int(config["run"]["seed"]))
    bank_hash = sha256_file(bank_path)
    signature = run_signature(config, bank_hash, __version__)
    output_root = Path(config["run"]["output_root"]).resolve()
    run_dir = output_root / f"CB4_{config['run']['phase']}_{signature[:12]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    selection_path = run_dir / "selected_tasks.csv"
    if not selection_path.exists():
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
        run_dir / "state.sqlite3", signature, {"manifest": manifest, "project_root": str(project_root)}
    )
    tasks = to_task_records(selected)
    total = len(tasks) * len(config["targets"]) * len(config["experiment"]["conditions"])
    completed = store.completed_episode_ids()
    report_config = config.get("reporting", {})
    reporter = LiveReporter(
        run_dir,
        total=total,
        phase=config["run"]["phase"],
        dry_run=bool(config["run"]["dry_run"]),
        refresh_seconds=float(report_config.get("refresh_seconds", 15)),
        heartbeat_seconds=float(report_config.get("heartbeat_seconds", 60)),
        baseline_completed=len(completed),
    )
    reporter.event(
        "run_started", phase=config["run"]["phase"], dry_run=bool(config["run"]["dry_run"]),
        total_episodes=total, resumed_completed=len(completed),
    )
    reporter.update(store.progress_snapshot(total), stage="starting", force=True)
    clients = RoleClients(config, project_id)
    checkpoint_every = int(config["run"].get("checkpoint_every_episodes", 1))
    newly_completed = 0

    try:
        for target_settings in config["targets"]:
            target_id = target_settings["id"]
            target_episode_ids = {
                _episode_id(signature, config["run"]["phase"], target_id, condition, task.assignment_id)
                for condition in config["experiment"]["conditions"] for task in tasks
            }
            if target_episode_ids.issubset(completed):
                reporter.event("target_skipped_complete", target_id=target_id)
                continue
            target = make_target(target_settings, bool(config["run"]["dry_run"]))
            try:
                reporter.event("target_load_started", target_id=target_id)
                with reporter.heartbeat(stage="loading_target", target_id=target_id):
                    target.load()
                reporter.event("target_loaded", target_id=target_id)
            except Exception as exc:
                store.record_failure(None, f"target_load:{target_id}", exc)
                reporter.event("target_load_failed", target_id=target_id, error_type=type(exc).__name__)
                reporter.update(store.progress_snapshot(total), stage="failed", force=True)
                raise

            try:
                for condition in config["experiment"]["conditions"]:
                    for task in tasks:
                        episode_id = _episode_id(
                            signature, config["run"]["phase"], target_id, condition, task.assignment_id
                        )
                        if store.episode_status(episode_id) == "complete":
                            continue
                        reporter.event(
                            "episode_started", episode_id=episode_id, assignment_id=task.assignment_id,
                            target_id=target_id, condition=condition,
                        )
                        reporter.update(
                            store.progress_snapshot(total), stage="running_episode",
                            current_target=target_id, current_condition=condition,
                            current_assignment=task.assignment_id, current_turn=0,
                        )

                        def on_turn(result: dict[str, Any]) -> None:
                            reporter.event(
                                "turn_completed", episode_id=result["episode_id"],
                                turn_index=result["turn_index"], success=result["success"],
                                response_class=result["final_response_class"],
                            )
                            reporter.update(
                                store.progress_snapshot(total), stage="running_episode",
                                current_target=target_id, current_condition=condition,
                                current_assignment=task.assignment_id,
                                current_turn=int(result["turn_index"]),
                                last_episode=result if result.get("terminal_reason") else None,
                            )

                        try:
                            result = run_episode(
                                episode_id=episode_id, condition=condition, task=task,
                                target_id=target_id, target=target, clients=clients,
                                store=store, config=config, on_turn=on_turn,
                            )
                            reporter.event(
                                "episode_completed", episode_id=episode_id, success=result["success"],
                                terminal_reason=result["terminal_reason"], queries_used=result["queries_used"],
                            )
                        except Exception as exc:
                            store.record_failure(episode_id, "episode", exc)
                            store.fail_episode(episode_id, exc)
                            result = {
                                "episode_id": episode_id, "assignment_id": task.assignment_id,
                                "target_id": target_id, "condition": condition, "queries_used": 0,
                                "success": False, "success_label": "NO",
                                "final_response_class": "technical_failure",
                                "chemistry_validation": "NOT_EVALUATED",
                                "terminal_reason": "technical_failure",
                            }
                            reporter.event(
                                "episode_failed", episode_id=episode_id, error_type=type(exc).__name__
                            )
                        newly_completed += 1
                        checkpoint_path: str | None = None
                        if newly_completed % checkpoint_every == 0:
                            snapshot_path = store.backup(run_dir / "checkpoint_snapshot.sqlite3")
                            sync_checkpoint_to_gcs(snapshot_path, config["run"].get("gcs_checkpoint_uri"))
                            checkpoint_path = str(snapshot_path)
                        reporter.update(
                            store.progress_snapshot(total), stage="running", current_target=target_id,
                            current_condition=condition, current_assignment=task.assignment_id,
                            current_turn=int(result["queries_used"]), last_episode=result,
                            checkpoint=checkpoint_path, force=True,
                        )
            finally:
                target.unload()
                reporter.event("target_unloaded", target_id=target_id)

        reporter.update(store.progress_snapshot(total), stage="exporting", force=True)
        snapshot_path = store.backup(run_dir / "checkpoint_snapshot.sqlite3")
        sync_checkpoint_to_gcs(snapshot_path, config["run"].get("gcs_checkpoint_uri"))
        export_results(
            store.path, selection_path, run_dir,
            int(config["experiment"]["target_query_budget"]),
            bool(config["run"].get("release_raw_outputs", False)),
        )
        reporter.event("run_completed", completed=store.progress_snapshot(total)["completed"])
        reporter.update(
            store.progress_snapshot(total), stage="complete", checkpoint=str(snapshot_path), force=True
        )
    except Exception as exc:
        reporter.event("run_failed", error_type=type(exc).__name__, error_message=str(exc)[:1000])
        reporter.update(store.progress_snapshot(total), stage="failed", force=True)
        raise
    finally:
        store.close()
    return run_dir
