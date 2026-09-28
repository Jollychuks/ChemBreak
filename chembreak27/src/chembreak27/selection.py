from __future__ import annotations
import hashlib, json
from pathlib import Path
from .constants import NAMESPACE, SELECTION_PROTOCOL, TASK_COUNT, ASSIGNMENT_IDS_SHA256
from .dataset import load_prompts, load_manifest, selected_tasks
from .utils import sha256_file

def _ids_sha(ids):
    return hashlib.sha256("\n".join(map(str, ids)).encode()).hexdigest()

def verify_bundle(prompts_path, manifest_path, lock_path):
    prompts = load_prompts(prompts_path)
    manifest = load_manifest(manifest_path)
    selected = selected_tasks(prompts_path, manifest_path)
    lock = json.loads(Path(lock_path).read_text())
    expected = {
        "namespace": NAMESPACE,
        "protocol": SELECTION_PROTOCOL,
        "task_count": TASK_COUNT,
        "assignment_ids_sha256": ASSIGNMENT_IDS_SHA256,
    }
    for k, v in expected.items():
        if lock.get(k) != v:
            raise RuntimeError(f"Lock metadata mismatch for {k}: {lock.get(k)!r} != {v!r}")
    if lock.get("source_prompts_sha256") != sha256_file(prompts_path):
        raise RuntimeError("Source prompt hash differs from lock")
    if lock.get("manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("Manifest hash differs from lock")
    ids = manifest.assignment_id.astype(str).tolist()
    if ids != lock.get("assignment_ids"):
        raise RuntimeError("Assignment IDs differ from lock")
    if _ids_sha(ids) != ASSIGNMENT_IDS_SHA256:
        raise RuntimeError("Assignment ID hash differs from lock")
    return {
        "status": "ok",
        "tasks": len(selected),
        "functional_categories": sorted(prompts.FunctionalCategory.astype(str).unique().tolist()),
        "semantic_categories": sorted(prompts.SemanticCategory.astype(str).unique().tolist()),
    }
