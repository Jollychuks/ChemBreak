from __future__ import annotations

import hashlib

from .constants import NAMESPACE
from .dataset import load_lock, selected_tasks
from .utils import sha256_file


def _ids_sha(ids):
    return hashlib.sha256("\n".join(map(str, ids)).encode()).hexdigest()


def verify_bundle(prompts_path, manifest_path, lock_path):
    lock = load_lock(lock_path)
    selected = selected_tasks(prompts_path, manifest_path, lock_path)
    if lock.get("namespace") != NAMESPACE:
        raise RuntimeError("Dataset lock namespace mismatch")
    if lock.get("source_prompts_sha256") != sha256_file(prompts_path):
        raise RuntimeError("Source prompt hash differs from lock")
    if lock.get("manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("Manifest hash differs from lock")
    ids = selected.assignment_id.astype(str).tolist()
    if ids != [str(x) for x in lock.get("assignment_ids", [])]:
        raise RuntimeError("Assignment IDs differ from lock")
    if _ids_sha(ids) != str(lock.get("assignment_ids_sha256")):
        raise RuntimeError("Assignment ID hash differs from lock")
    return {
        "status": "ok",
        "dataset_profile": str(lock["dataset_profile"]),
        "protocol": str(lock["protocol"]),
        "tasks": len(selected),
        "functional_categories": sorted(selected.functional_category.astype(str).unique().tolist()),
        "semantic_categories": sorted(selected.semantic_category.astype(str).unique().tolist()),
    }
