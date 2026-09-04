from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeRemainingColumn,
    )
except ImportError:  # Minimal fallback for schema tests before dependencies are installed.
    class Console:  # type: ignore[no-redef]
        def print(self, *values, **kwargs):
            print(*values)

    class _Column:
        def __init__(self, *args, **kwargs):
            pass

    BarColumn = MofNCompleteColumn = SpinnerColumn = TextColumn = TimeRemainingColumn = _Column

    class Progress:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            self.completed = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def add_task(self, description, total, completed=0):
            self.completed = completed
            return 0

        def update(self, task_id, **kwargs):
            return None

        def advance(self, task_id, advance=1):
            self.completed += advance

from . import __version__
from .benchmark import load_and_validate_task_bank, select_tasks, to_task_records, write_selection
from .checkpoint import CheckpointStore, sync_checkpoint_to_gcs
from .conditions import run_episode
from .config import load_config, resolve_project_path, run_signature, validate_config
from .metrics import export_results
from .providers import RoleClients
from .storage import configure_content_storage, verify_content_storage
from .targets import make_target
from .utils import require_live_gate, set_global_seed, sha256_file, stable_id, utc_now


console = Console()


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
        "namespace": "CB2",
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
        f"CB2-{phase}-{target_id}-{condition}-{assignment_id}-"
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
    run_dir = output_root / f"CB2_{config['run']['phase']}_{signature[:12]}"
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
        run_dir / "state.sqlite3",
        signature,
        {"manifest": manifest, "project_root": str(project_root)},
    )
    clients = RoleClients(config, project_id)
    tasks = to_task_records(selected)
    completed = store.completed_episode_ids()
    total = len(tasks) * len(config["targets"]) * len(config["experiment"]["conditions"])
    checkpoint_every = int(config["run"].get("checkpoint_every_episodes", 1))
    newly_completed = 0
    console.print(
        f"[bold]ChemBreak CB2[/bold] phase={config['run']['phase']} tasks={len(tasks)} "
        f"targets={len(config['targets'])} conditions=4 episodes={total}"
    )
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    try:
        with progress:
            progress_id = progress.add_task("episodes", total=total, completed=len(completed))
            for target_settings in config["targets"]:
                target_id = target_settings["id"]
                target_episode_ids = {
                    _episode_id(
                        signature,
                        config["run"]["phase"],
                        target_id,
                        condition,
                        task.assignment_id,
                    )
                    for condition in config["experiment"]["conditions"]
                    for task in tasks
                }
                if target_episode_ids.issubset(completed):
                    continue
                progress.update(progress_id, description=f"loading {target_id}")
                target = make_target(target_settings, bool(config["run"]["dry_run"]))
                try:
                    target.load()
                except Exception as exc:
                    store.record_failure(None, f"target_load:{target_id}", exc)
                    raise
                try:
                    for condition in config["experiment"]["conditions"]:
                        for task in tasks:
                            episode_id = _episode_id(
                                signature,
                                config["run"]["phase"],
                                target_id,
                                condition,
                                task.assignment_id,
                            )
                            if store.episode_status(episode_id) == "complete":
                                continue
                            progress.update(
                                progress_id,
                                description=f"{target_id} {condition} {task.assignment_id}",
                            )
                            try:
                                run_episode(
                                    episode_id=episode_id,
                                    condition=condition,
                                    task=task,
                                    target_id=target_id,
                                    target=target,
                                    clients=clients,
                                    store=store,
                                    config=config,
                                )
                            except Exception as exc:
                                store.record_failure(episode_id, "episode", exc)
                                store.fail_episode(episode_id, exc)
                                console.print(f"[red]Episode failed[/red] {episode_id}: {exc}")
                            newly_completed += 1
                            progress.advance(progress_id)
                            if newly_completed % checkpoint_every == 0:
                                snapshot = store.backup(run_dir / "checkpoint_snapshot.sqlite3")
                                sync_checkpoint_to_gcs(snapshot, config["run"].get("gcs_checkpoint_uri"))
                finally:
                    target.unload()
        snapshot = store.backup(run_dir / "checkpoint_snapshot.sqlite3")
        sync_checkpoint_to_gcs(snapshot, config["run"].get("gcs_checkpoint_uri"))
        export_results(
            store.path,
            selection_path,
            run_dir,
            int(config["experiment"]["target_query_budget"]),
            bool(config["run"].get("release_raw_outputs", False)),
        )
    finally:
        store.close()
    console.print(f"[green]Run complete[/green] {run_dir}")
    return run_dir
