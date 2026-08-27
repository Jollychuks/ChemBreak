from __future__ import annotations
import argparse
import json
from pathlib import Path
import tempfile

from google.cloud import storage

from chembreak.config import load_config
from chembreak.gcs import split_gs, sync_down_prefix, sync_up_dir
from chembreak.runner import _dirs, _load_selected
from chembreak.store import append_jsonl, completed_keys, load_jsonl


def _download_optional(uri: str, dest: Path, project: str) -> bool:
    bucket_name, blob_name = split_gs(uri)
    client = storage.Client(project=project or None)
    blob = client.bucket(bucket_name).blob(blob_name)
    if not blob.exists(client):
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(dest))
    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Import only completed frozen attack assets from an earlier compatible ChemBreak run."
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--previous-gcs-output-uri", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    base, restricted, public = _dirs(cfg)
    base.mkdir(parents=True, exist_ok=True)
    current_uri = str(cfg.get("gcp", {}).get("gcs_output_uri", "")).rstrip("/")
    previous_uri = args.previous_gcs_output_uri.rstrip("/")

    if not previous_uri or previous_uri == current_uri:
        print("[IMPORT] Previous output URI is empty or equals the current output URI. Nothing imported.")
        return

    if current_uri:
        sync_down_prefix(current_uri, base)

    tasks, _ = _load_selected(cfg, base)
    selected = {str(t["task_id"]) for t in tasks}
    dest = restricted / "attack_assets.jsonl"
    existing = completed_keys(dest, ("task_id",))

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "attack_assets.jsonl"
        source_uri = previous_uri + "/restricted/attack_assets.jsonl"
        if not _download_optional(source_uri, src, str(cfg.get("gcp", {}).get("project", ""))):
            print(f"[IMPORT] No previous frozen asset file found at {source_uri}. Continuing without import.")
            return
        rows = load_jsonl(src)

    imported = 0
    ignored = 0
    for row in rows:
        task_id = str(row.get("task_id", ""))
        if row.get("status") != "complete" or not task_id or task_id not in selected:
            ignored += 1
            continue
        if (task_id,) in existing:
            ignored += 1
            continue
        assets = row.get("assets")
        if not isinstance(assets, dict):
            ignored += 1
            continue
        append_jsonl(dest, {
            "status": "complete",
            "run_id": cfg.get("run_id", "default"),
            "task_id": task_id,
            "assets": assets,
            "imported_from": source_uri,
            "source_run_id": row.get("run_id"),
        })
        existing.add((task_id,))
        imported += 1

    public.mkdir(parents=True, exist_ok=True)
    (public / "asset_import_summary.json").write_text(json.dumps({
        "previous_output_uri": previous_uri,
        "selected_tasks": len(selected),
        "imported_complete_assets": imported,
        "ignored_rows": ignored,
        "complete_assets_available_after_import": len(existing & {(x,) for x in selected}),
    }, indent=2), encoding="utf-8")

    if current_uri:
        sync_up_dir(base, current_uri)
    print(f"[IMPORT] Imported {imported} completed frozen assets from the previous compatible run.")
    print(f"[IMPORT] Current selected tasks already complete after import: {len(existing & {(x,) for x in selected})}/{len(selected)}")


if __name__ == "__main__":
    main()
