from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any


ENVIRONMENT_PATHS = {
    "hf_home": "HF_HOME",
    "hf_hub_cache": "HF_HUB_CACHE",
    "hf_modules_cache": "HF_MODULES_CACHE",
    "xdg_cache_home": "XDG_CACHE_HOME",
    "torch_home": "TORCH_HOME",
    "torchinductor_cache": "TORCHINDUCTOR_CACHE_DIR",
    "triton_cache": "TRITON_CACHE_DIR",
    "cuda_cache": "CUDA_CACHE_PATH",
    "pip_cache": "PIP_CACHE_DIR",
    "temp_dir": "TMPDIR",
}


def _resolved(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def is_within(path: str | Path, parent: str | Path) -> bool:
    try:
        _resolved(path).relative_to(_resolved(parent))
        return True
    except ValueError:
        return False


def configure_content_storage(config: dict[str, Any]) -> dict[str, str]:
    storage = config["storage"]
    paths: dict[str, str] = {}
    for key in (
        "storage_root",
        *ENVIRONMENT_PATHS,
        "python_packages",
        "offload_dir",
        "preflight_dir",
    ):
        path = _resolved(storage[key])
        path.mkdir(parents=True, exist_ok=True)
        paths[key] = str(path)

    for key, variable in ENVIRONMENT_PATHS.items():
        os.environ[variable] = paths[key]
    os.environ["TRANSFORMERS_CACHE"] = paths["hf_hub_cache"]
    os.environ["HF_DATASETS_CACHE"] = str(_resolved(Path(paths["hf_home"]) / "datasets"))
    os.environ["NUMBA_CACHE_DIR"] = str(_resolved(Path(paths["storage_root"]) / "cache" / "numba"))
    os.environ["PYTORCH_KERNEL_CACHE_PATH"] = str(
        _resolved(Path(paths["storage_root"]) / "cache" / "torch_kernels")
    )
    os.environ["MPLCONFIGDIR"] = str(
        _resolved(Path(paths["storage_root"]) / "cache" / "matplotlib")
    )
    os.environ["PYTHONPYCACHEPREFIX"] = str(
        _resolved(Path(paths["storage_root"]) / "cache" / "python_bytecode")
    )
    os.environ["TMP"] = paths["temp_dir"]
    os.environ["TEMP"] = paths["temp_dir"]
    os.environ["CHEMBREAK_OFFLOAD_DIR"] = paths["offload_dir"]
    os.environ["CHEMBREAK_PYTHON_PACKAGES"] = paths["python_packages"]
    Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["PYTORCH_KERNEL_CACHE_PATH"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["PYTHONPYCACHEPREFIX"]).mkdir(parents=True, exist_ok=True)
    return paths


def verify_content_storage(
    config: dict[str, Any], project_root: str | Path
) -> dict[str, Any]:
    storage = config["storage"]
    content_root = _resolved(storage["content_root"])
    if not content_root.is_dir():
        raise RuntimeError(f"Configured content disk is unavailable: {content_root}")
    if bool(storage.get("require_separate_mount", True)):
        if content_root.stat().st_dev == Path("/").stat().st_dev:
            raise RuntimeError(
                f"{content_root} is not a separate mounted filesystem. Refusing to load models."
            )

    required_paths = {
        "project_root": _resolved(project_root),
        "output_root": _resolved(config["run"]["output_root"]),
        "storage_root": _resolved(storage["storage_root"]),
        "python_packages": _resolved(storage["python_packages"]),
        **{key: _resolved(storage[key]) for key in ENVIRONMENT_PATHS},
        "offload_dir": _resolved(storage["offload_dir"]),
        "preflight_dir": _resolved(storage["preflight_dir"]),
    }
    target_paths: dict[str, Path] = {}
    for target in config["targets"]:
        target_paths[f"target:{target['id']}:cache_dir"] = _resolved(target["cache_dir"])
        target_paths[f"target:{target['id']}:offload_folder"] = _resolved(
            target["offload_folder"]
        )
    required_paths.update(target_paths)
    outside = {key: str(path) for key, path in required_paths.items() if not is_within(path, content_root)}
    if outside and bool(storage.get("require_content_routing", True)):
        details = ", ".join(f"{key}={value}" for key, value in sorted(outside.items()))
        raise RuntimeError(f"Large-write paths must stay under {content_root}: {details}")

    for path in required_paths.values():
        path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(content_root)
    gib = 1024**3
    free_gb = usage.free / gib
    minimum = float(storage.get("minimum_free_gb", 0))
    if free_gb < minimum:
        raise RuntimeError(
            f"Content disk has {free_gb:.1f} GiB free, below the required {minimum:.1f} GiB."
        )
    return {
        "content_root": str(content_root),
        "separate_mount": content_root.stat().st_dev != Path("/").stat().st_dev,
        "total_gib": round(usage.total / gib, 2),
        "used_gib": round(usage.used / gib, 2),
        "free_gib": round(free_gb, 2),
        "minimum_free_gib": minimum,
        "routed_paths": {key: str(value) for key, value in sorted(required_paths.items())},
    }
