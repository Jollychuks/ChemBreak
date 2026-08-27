from __future__ import annotations
from pathlib import Path
import os
import shutil
from typing import Any


def resolved_hf_cache_dir(runtime_cfg: dict[str, Any] | None = None) -> Path:
    runtime_cfg = runtime_cfg or {}
    configured = str(runtime_cfg.get("hf_cache_dir", "") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    # In Colab Enterprise, /content is a large attached data disk. Prefer it when present.
    if Path("/content").exists():
        return Path("/content/hf_cache")
    return (Path.home() / ".cache" / "huggingface").resolve()


def configure_cache_environment(runtime_cfg: dict[str, Any] | None = None) -> Path:
    cache = resolved_hf_cache_dir(runtime_cfg)
    cache.mkdir(parents=True, exist_ok=True)
    xet = cache / "xet"
    xet.mkdir(parents=True, exist_ok=True)
    torch_cache = cache.parent / "torch_cache"
    torch_cache.mkdir(parents=True, exist_ok=True)
    tmp = cache.parent / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    # Explicitly route every common model/cache path away from the small system disk.
    os.environ["HF_HOME"] = str(cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache / "hub")
    os.environ["HF_HUB_CACHE"] = str(cache / "hub")
    os.environ["HF_XET_CACHE"] = str(xet)
    os.environ["TORCH_HOME"] = str(torch_cache)
    os.environ["TMPDIR"] = str(tmp)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return cache


def disk_snapshot(path: Path) -> dict[str, float | str]:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    gb = 2**30
    return {
        "path": str(path),
        "total_gb": round(usage.total / gb, 2),
        "used_gb": round(usage.used / gb, 2),
        "free_gb": round(usage.free / gb, 2),
    }


def legacy_system_hf_cache() -> Path:
    return (Path.home() / ".cache" / "huggingface").resolve()


def cleanup_legacy_cache_if_requested(runtime_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    runtime_cfg = runtime_cfg or {}
    enabled = bool(runtime_cfg.get("cleanup_legacy_system_hf_cache", False))
    cache = resolved_hf_cache_dir(runtime_cfg)
    legacy = legacy_system_hf_cache()
    result = {"enabled": enabled, "legacy_path": str(legacy), "removed": False}
    if not enabled or not legacy.exists():
        return result
    try:
        if legacy.samefile(cache):
            result["reason"] = "legacy path is the configured cache"
            return result
    except Exception:
        if str(legacy) == str(cache):
            result["reason"] = "legacy path is the configured cache"
            return result
    shutil.rmtree(legacy, ignore_errors=False)
    result["removed"] = True
    return result
