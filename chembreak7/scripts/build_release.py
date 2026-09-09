from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT.parent / "chembreak7_Google_Cloud_Enterprise_Ready.zip"
EXCLUDED_PARTS = {".venv", ".pytest_cache", ".ruff_cache", "__pycache__"}
EXCLUDED_NAMES = {"uv.lock", "PACKAGE_MANIFEST.sha256"}


def included_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts)
        and not any(part.endswith(".egg-info") for part in path.relative_to(ROOT).parts)
        and path.name not in EXCLUDED_NAMES
        and path.suffix != ".pyc"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    files = included_files()
    manifest = "".join(
        f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in files
    )
    manifest_path = ROOT / "PACKAGE_MANIFEST.sha256"
    manifest_path.write_text(manifest, encoding="utf-8")
    files.append(manifest_path)
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(files):
            archive.write(path, Path("chembreak7") / path.relative_to(ROOT))
    print(OUTPUT)
    print(sha256(OUTPUT))


if __name__ == "__main__":
    main()
