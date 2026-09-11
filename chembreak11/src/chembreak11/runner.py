from __future__ import annotations

import json
import os
import platform
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from . import CHECKPOINT_PROTOCOL_VERSION, __version__
from .benchmark import (
    load_and_validate_task_bank,
    select_phase_tasks,
    to_task_records,
    write_selection,
)
from .checkpoint import CheckpointStore, sync_checkpoint_to_gcs
from .conditions import recover_pending_turn, run_episode
from .config import load_config, project_root, run_signature, validate_config
from .metrics import export_results
from .policy import AdaptiveQPolicy
from .providers import RoleClients
from .reporting import LiveReporter
from .schema import CONDITIONS, TaskRecord
from .storage import configure_content_storage, verify_content_storage
from .targets import make_target
from .utils import require_live_gate, set_global_seed, sha256_file, stable_id, utc_now


@dataclass(slots=True)
class RunContext:
    config: dict[str, Any]
    root: Path
    signature: str
    run_dir: Path
    selection_path: Path
    tasks: list[TaskRecord]
    store: CheckpointStore
    project_id: str | None


def _project_id(config: dict[str, Any]) -> str | None:
    if config["run"]["dry_run"]:
        return None
    value = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT_ID")
    if not value:
        raise RuntimeError("Set GOOGLE_CLOUD_PROJECT to the active project ID.")
    return value


def _episode_id(
    signature: str, phase: str, condition: str, target_id: str, assignment_id: str,
) -> str:
    return (
        f"CB11-{phase}-{target_id}-{condition}-{assignment_id}-"
        f"{stable_id(signature, condition, target_id, assignment_id, length=8)}"
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
    bank_path = bank_setting if bank_setting.is_absolute() else root / bank_setting
    frame = load_and_validate_task_bank(bank_path)
    selected = select_phase_tasks(frame, config["run"]["phase"])
    bank_hash = sha256_file(bank_path)
    policy_digest = None
    if config["policy"]["mode"] == "frozen":
        policy_path = Path(config["policy"]["artifact_path"]).resolve()
        if not policy_path.is_file():
            raise FileNotFoundError(
                f"Frozen policy artifact is missing: {policy_path}. Complete development and freeze it first."
            )
        policy_digest = sha256_file(policy_path)
    signature = run_signature(
        config, bank_hash, CHECKPOINT_PROTOCOL_VERSION, policy_sha256=policy_digest,
    )
    run_dir = Path(config["run"]["output_root"]).resolve() / f"CB11_{config['run']['phase']}_{signature[:12]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    selection_path = run_dir / "selected_tasks.csv"
    if selection_path.exists():
        frozen = pd.read_csv(selection_path)
        if list(frozen.assignment_id.astype(str)) != list(selected.assignment_id.astype(str)):
            raise RuntimeError("The existing run has a different frozen task selection.")
    else:
        write_selection(selected, selection_path)
    project_id = _project_id(config)
    manifest = {
        "namespace": "CB11", "software_version": __version__,
        "checkpoint_protocol_version": CHECKPOINT_PROTOCOL_VERSION,
        "run_signature": signature, "task_bank_sha256": bank_hash,
        "created_at_utc": utc_now(), "phase": config["run"]["phase"],
        "dry_run": config["run"]["dry_run"],
        "conditions": list(config["experiment"]["conditions"]),
        "project_id_recorded": bool(project_id), "policy_mode": config["policy"]["mode"],
        "policy_sha256": policy_digest, "storage": storage_report,
        "python": sys.version, "platform": platform.platform(),
        "adaptive_definition": "Verified success on turn 2 or later after target feedback.",
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
    }
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("run_signature") != signature:
            raise RuntimeError("Existing run manifest has a different signature.")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    store = CheckpointStore(run_dir / "state.sqlite3", signature, {"manifest": manifest})
    return RunContext(config, root, signature, run_dir, selection_path, to_task_records(selected), store, project_id)


def _export(context: RunContext) -> dict[str, str]:
    context.store.backup(context.run_dir / "checkpoint_snapshot.sqlite3")
    manifest = export_results(
        context.store.path, context.selection_path, context.run_dir,
        int(context.config["experiment"]["target_query_budget"]),
        bool(context.config["run"].get("release_raw_outputs", False)),
    )
    policy = AdaptiveQPolicy(context.config["policy"], int(context.config["run"]["seed"]))
    policy_metrics = policy.export_metrics(context.run_dir / "release" / "policy_q_values.csv")
    manifest["policy_q_values"] = str(policy_metrics)
    (context.run_dir / "release" / "policy_summary.json").write_text(
        json.dumps(policy.summary(), indent=2, sort_keys=True), encoding="utf-8",
    )
    return manifest


def run_condition_target(config_path: str | Path, condition: str, target_id: str) -> Path:
    context = _initialize(config_path)
    config = context.config
    if condition not in config["experiment"]["conditions"]:
        context.store.close()
        raise ValueError(f"Unknown or disabled condition: {condition}")
    settings = next((item for item in config["targets"] if item["id"] == target_id), None)
    if settings is None:
        context.store.close()
        raise ValueError(f"Unknown target: {target_id}")
    total = len(context.tasks) * len(config["targets"]) * len(config["experiment"]["conditions"])
    report = config.get("reporting", {})
    reporter = LiveReporter(
        context.run_dir, total=total, phase=config["run"]["phase"], condition=condition,
        target_scope=target_id, dry_run=bool(config["run"]["dry_run"]),
        refresh_seconds=float(report.get("refresh_seconds", 15)),
        heartbeat_seconds=float(report.get("heartbeat_seconds", 60)),
    )
    clients = RoleClients(config, context.project_id)
    learned_policy = AdaptiveQPolicy(
        config["policy"], int(config["run"]["seed"]) + int(stable_id(target_id, length=6), 16),
    )
    target = make_target(settings, bool(config["run"]["dry_run"]))
    try:
        reporter.update(context.store.progress_snapshot(total), stage="starting", current_target=target_id, force=True)
        expected_ids = {
            _episode_id(context.signature, config["run"]["phase"], condition, target_id, task.assignment_id)
            for task in context.tasks
        }
        if expected_ids.issubset(context.store.completed_episode_ids(target_id)):
            reporter.update(context.store.progress_snapshot(total), stage="target_already_complete", force=True)
            _export(context)
            return context.run_dir
        try:
            with reporter.heartbeat(stage="loading_target", target_id=target_id):
                target.load()
        except Exception as exc:
            context.store.record_failure(None, f"target_load:{target_id}", exc)
            reporter.event("target_load_failed", target_id=target_id, error_type=type(exc).__name__)
            _export(context)
            raise
        for task in context.tasks:
            episode_id = _episode_id(
                context.signature, config["run"]["phase"], condition, target_id, task.assignment_id,
            )
            if context.store.episode_status(episode_id) == "complete":
                continue
            reporter.update(
                context.store.progress_snapshot(total), stage="running_episode",
                current_target=target_id, current_assignment=task.assignment_id, force=True,
            )
            try:
                result = run_episode(
                    episode_id=episode_id, task=task, target_id=target_id, target=target,
                    condition=condition, clients=clients, store=context.store, config=config,
                    learned_policy=learned_policy,
                    on_turn=lambda value, assignment_id=task.assignment_id: reporter.update(
                        context.store.progress_snapshot(total), stage="turn_checkpointed",
                        current_target=target_id, current_assignment=assignment_id,
                        current_turn=value["turn_index"],
                    ),
                )
                reporter.event(
                    "episode_complete", episode_id=episode_id, assignment_id=task.assignment_id,
                    target_id=target_id, queries=result["queries_used"], condition=condition,
                    verified_success=result["verified_success"],
                    bootstrap_success=result["bootstrap_success"], adaptive_success=result["adaptive_success"],
                )
                reporter.update(
                    context.store.progress_snapshot(total), stage="episode_complete",
                    last_episode=result, current_turn=result["turn_index"], force=True,
                )
            except Exception as exc:  # noqa: BLE001
                reporter.event(
                    "episode_paused", episode_id=episode_id, assignment_id=task.assignment_id,
                    target_id=target_id, error_type=type(exc).__name__,
                )
                reporter.update(context.store.progress_snapshot(total), stage="episode_paused_continuing", force=True)
            _export(context)
            sync_checkpoint_to_gcs(
                context.run_dir / "checkpoint_snapshot.sqlite3", config["run"].get("gcs_checkpoint_uri"),
            )
        snapshot = context.store.progress_snapshot(total)
        target_complete = expected_ids.issubset(context.store.completed_episode_ids(target_id))
        reporter.update(snapshot, stage="target_complete" if target_complete else "target_incomplete", force=True)
        _export(context)
        return context.run_dir
    finally:
        target.unload()
        context.store.close()


def run_target(config_path: str | Path, target_id: str) -> Path:
    return run_condition_target(config_path, "C3_ADAPTIVE_MDP", target_id)


def run_condition(config_path: str | Path, condition: str) -> Path:
    config = load_config(config_path)
    target_ids = [item["id"] for item in config["targets"]]
    path = None
    for target_id in target_ids:
        path = run_condition_target(config_path, condition, target_id)
    assert path is not None
    return path


def run_all_targets(config_path: str | Path) -> Path:
    return run_condition(config_path, "C3_ADAPTIVE_MDP")


def run_all_conditions(config_path: str | Path) -> Path:
    path = None
    for condition in CONDITIONS:
        path = run_condition(config_path, condition)
    assert path is not None
    return path


def recover_pending(config_path: str | Path, target_id: str | None = None) -> dict[str, Any]:
    context = _initialize(config_path)
    clients = RoleClients(context.config, context.project_id)
    learned_policy = AdaptiveQPolicy(context.config["policy"], int(context.config["run"]["seed"]))
    tasks = {task.assignment_id: task for task in context.tasks}
    attempted = recovered = failed = 0
    try:
        for episode in context.store.pending_episodes(target_id):
            pending = context.store.pending_stage(str(episode["episode_id"]))
            if pending is None:
                continue
            attempted += 1
            try:
                result = recover_pending_turn(
                    store=context.store, clients=clients, config=context.config,
                    task=tasks[str(episode["assignment_id"])], target_id=str(episode["target_id"]),
                    condition=str(episode["condition"]), episode_id=str(episode["episode_id"]),
                    learned_policy=learned_policy,
                )
                recovered += int(result is not None)
            except Exception:  # noqa: BLE001
                failed += 1
        manifest = _export(context)
        return {"run_dir": str(context.run_dir), "attempted": attempted, "recovered": recovered, "failed": failed, "exports": manifest}
    finally:
        context.store.close()


def strict_completion_gate(
    config_path: str | Path, condition: str | None = None,
) -> dict[str, Any]:
    context = _initialize(config_path)
    try:
        if condition is not None and condition not in CONDITIONS:
            raise ValueError(f"Unknown condition: {condition}")
        multiplier = 1 if condition else len(context.config["experiment"]["conditions"])
        expected = len(context.tasks) * len(context.config["targets"]) * multiplier
        with sqlite3.connect(context.store.path) as connection:
            if condition:
                statuses = dict(connection.execute(
                    "SELECT status,COUNT(*) FROM episodes WHERE condition=? GROUP BY status", (condition,),
                ).fetchall())
            else:
                statuses = dict(connection.execute("SELECT status,COUNT(*) FROM episodes GROUP BY status").fetchall())
        completed = int(statuses.get("complete", 0))
        if completed != expected or any(key != "complete" for key in statuses):
            raise RuntimeError(
                f"Run is incomplete: expected {expected} complete episodes, found {statuses}. "
                "Run recovery, then rerun any target cell that remains incomplete."
            )
        exports = _export(context)
        return {
            "status": "complete", "condition": condition or "ALL_CONDITIONS",
            "episodes": completed, "run_dir": str(context.run_dir), "exports": exports,
        }
    finally:
        context.store.close()
