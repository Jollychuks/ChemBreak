from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "RELEASE_MANIFEST.json"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == OUTPUT:
            continue
        relative = path.relative_to(ROOT)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        files.append({
            "path": relative.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_hash(path),
        })

    datasets = {}
    for profile, lock_name in (
        ("prompts28", "prompts28_lock.json"),
        ("chemharm500", "chemharm500_lock.json"),
    ):
        lock = json.loads((ROOT / "data" / lock_name).read_text())
        datasets[profile] = {
            "task_count": int(lock["task_count"]),
            "protocol": str(lock["protocol"]),
            "source_prompts_sha256": str(lock["source_prompts_sha256"]),
            "manifest_sha256": str(lock["manifest_sha256"]),
        }

    manifest = {
        "project": "ChemBreak",
        "package_version": "1.5.0",
        "dataset_profiles": datasets,
        "file_count_excluding_manifest": len(files),
        "files": files,
    }
    OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"files": len(files), "output": str(OUTPUT)}))


if __name__ == "__main__":
    main()
