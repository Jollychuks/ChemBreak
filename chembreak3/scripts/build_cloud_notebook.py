from __future__ import annotations

import json
from pathlib import Path


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    markdown(
        """# ChemBreak3 on Google Cloud Notebook Enterprise

This notebook runs the frozen ChemHarm task bank while keeping the repository, model downloads, caches, temporary files, offload data, checkpoints, and outputs on the large disk mounted at `/content`.

Before starting, stop any still-running ChemBreak V4 model-loading cell. This notebook does not delete the old cache from `/home/jupyter`; it prevents new CB3 writes there.
"""
    ),
    code(
        """# EDIT THIS CELL
GOOGLE_CLOUD_PROJECT = "YOUR_GOOGLE_CLOUD_PROJECT_ID"
PHASE = "test"  # test, pilot, or production
LIVE = False  # keep False for the first complete pass
GCS_CHECKPOINT_URI = None  # example: gs://private-bucket/chembreak/chembreak3
REPO_URL = "https://github.com/Jollychuks/ChemBreak.git"
BRANCH = "main"
PROJECT_SUBDIR = "chembreak3"

CONTENT_ROOT = "/content"
STORAGE_ROOT = "/content/chembreak3_storage"
MINIMUM_FREE_GIB = 100
"""
    ),
    markdown("## 1. Route all heavyweight writes to `/content`"),
    code(
        """from pathlib import Path
import os
import shutil
import subprocess
import sys

content_root = Path(CONTENT_ROOT).resolve()
storage_root = Path(STORAGE_ROOT).resolve()
assert content_root.is_dir(), f"Mounted content disk not found: {content_root}"
assert content_root.stat().st_dev != Path("/").stat().st_dev, (
    f"{content_root} is not a separate filesystem. Stop before loading models."
)

paths = {
    "HF_HOME": storage_root / "cache" / "huggingface",
    "HF_HUB_CACHE": storage_root / "cache" / "huggingface" / "hub",
    "TRANSFORMERS_CACHE": storage_root / "cache" / "huggingface" / "hub",
    "HF_MODULES_CACHE": storage_root / "cache" / "huggingface" / "modules",
    "HF_DATASETS_CACHE": storage_root / "cache" / "huggingface" / "datasets",
    "XDG_CACHE_HOME": storage_root / "cache" / "xdg",
    "TORCH_HOME": storage_root / "cache" / "torch",
    "TORCHINDUCTOR_CACHE_DIR": storage_root / "cache" / "torchinductor",
    "TRITON_CACHE_DIR": storage_root / "cache" / "triton",
    "CUDA_CACHE_PATH": storage_root / "cache" / "cuda",
    "PYTORCH_KERNEL_CACHE_PATH": storage_root / "cache" / "torch_kernels",
    "NUMBA_CACHE_DIR": storage_root / "cache" / "numba",
    "MPLCONFIGDIR": storage_root / "cache" / "matplotlib",
    "PIP_CACHE_DIR": storage_root / "cache" / "pip",
    "PYTHONPYCACHEPREFIX": storage_root / "cache" / "python_bytecode",
    "TMPDIR": storage_root / "tmp",
    "TMP": storage_root / "tmp",
    "TEMP": storage_root / "tmp",
    "CHEMBREAK_OFFLOAD_DIR": storage_root / "offload",
}
for variable, path in paths.items():
    path.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(path)

usage = shutil.disk_usage(content_root)
free_gib = usage.free / 1024**3
assert free_gib >= MINIMUM_FREE_GIB, (
    f"Only {free_gib:.1f} GiB is free on {content_root}; {MINIMUM_FREE_GIB} GiB is required."
)
subprocess.run(["df", "-h", "/", str(content_root)], check=True)
print(f"CB3 storage root: {storage_root}")
print(f"Free on content disk: {free_gib:.1f} GiB")

legacy_cache = Path("/home/jupyter/.cache/huggingface")
if legacy_cache.exists():
    subprocess.run(["du", "-sh", str(legacy_cache)], check=False)
    print("Legacy V4 cache detected above. It is not used or deleted by chembreak3.")
"""
    ),
    markdown(
        """## 2. Clone or reuse the repository on `/content`

The cell clones only when the checkout is absent. If it already exists, it prints the current commit and leaves local work untouched. Use your normal Git workflow to update it.
"""
    ),
    code(
        """checkout = content_root / "chembreak_repo"
if not checkout.exists():
    subprocess.run(
        ["git", "clone", "--branch", BRANCH, "--single-branch", REPO_URL, str(checkout)],
        check=True,
    )
else:
    assert (checkout / ".git").is_dir(), f"Existing path is not a Git checkout: {checkout}"
    print("Using existing checkout without modifying it:", checkout)
    subprocess.run(["git", "-C", str(checkout), "status", "--short", "--branch"], check=True)

PROJECT_DIR = (checkout / PROJECT_SUBDIR).resolve()
assert PROJECT_DIR.is_relative_to(content_root), PROJECT_DIR
assert (PROJECT_DIR / "pyproject.toml").exists(), (
    f"Project not found: {PROJECT_DIR}. Add the chembreak3 folder to the repository first."
)
os.chdir(PROJECT_DIR)
print("Project:", PROJECT_DIR)
"""
    ),
    markdown(
        """## 3. Preserve the Notebook Enterprise Torch stack

The notebook image already supplies a matched CUDA, Torch, and Torchvision stack. This cell never installs or upgrades those packages. It adds only missing non-Torch dependencies under `/content`, then verifies the active package origins before preflight.

Restart the kernel before running this notebook if a ChemBreak1 or ChemBreak2 installation cell has already run in the current kernel.
"""
    ),
    code(
        """import importlib
import site

contaminated_modules = []
for module_name in ("torch", "torchvision", "transformers", "peft"):
    module = sys.modules.get(module_name)
    origin = str(getattr(module, "__file__", "")) if module else ""
    if "/content/chembreak1_storage/" in origin or "/content/chembreak2_storage/" in origin:
        contaminated_modules.append(f"{module_name}={origin}")
if contaminated_modules:
    raise RuntimeError(
        "This kernel still has packages loaded by an earlier ChemBreak version. "
        "Restart the kernel, open the ChemBreak3 notebook, and run from the first cell. "
        + "; ".join(contaminated_modules)
    )

package_dir = storage_root / "python_packages"
package_dir.mkdir(parents=True, exist_ok=True)
for forbidden in ("torch", "torchvision", "torchaudio", "torchgen", "functorch"):
    if (package_dir / forbidden).exists():
        raise RuntimeError(
            f"Unexpected {forbidden} package found in {package_dir}. "
            "Use a fresh chembreak3_storage directory before continuing."
        )

required = {
    "accelerate": "accelerate>=1.2,<2",
    "google.auth": "google-auth>=2.35,<3",
    "google.cloud.storage": "google-cloud-storage>=2.18,<4",
    "google.genai": "google-genai>=1.47,<2",
    "openai": "openai>=1.57,<3",
    "numpy": "numpy>=1.26,<3",
    "pandas": "pandas>=2.2,<3",
    "peft": "peft>=0.13,<1",
    "pydantic": "pydantic>=2.9,<3",
    "yaml": "PyYAML>=6.0,<7",
    "rdkit": "rdkit>=2024.3",
    "rich": "rich>=13.9,<15",
    "scipy": "scipy>=1.13,<2",
    "sentencepiece": "sentencepiece>=0.2,<1",
    "tenacity": "tenacity>=9,<10",
    "transformers": "transformers>=4.46,<5",
}

missing_specs = []
for module_name, requirement in required.items():
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError:
        missing_specs.append(requirement)

if missing_specs:
    print("Installing missing non-Torch packages on /content:", missing_specs)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            str(package_dir),
            "--cache-dir",
            str(paths["PIP_CACHE_DIR"]),
            "--no-deps",
            *missing_specs,
        ],
        check=True,
    )

site.addsitedir(str(package_dir))
sys.path.insert(0, str(package_dir))
sys.path.insert(0, str(PROJECT_DIR / "src"))
existing_pythonpath = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = os.pathsep.join(
    value for value in (str(PROJECT_DIR / "src"), str(package_dir), existing_pythonpath) if value
)
importlib.invalidate_caches()

failed_imports = {}
for module_name in required:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        failed_imports[module_name] = f"{type(exc).__name__}: {exc}"
if failed_imports:
    raise RuntimeError(f"Dependency verification failed: {failed_imports}")

import torch
import torchvision
import transformers
import peft

torch_origin = str(Path(torch.__file__).resolve())
assert not torch_origin.startswith(str(storage_root)), (
    f"Torch must come from the Notebook Enterprise image, not {storage_root}: {torch_origin}"
)
print("torch:", torch.__version__, torch_origin)
print("torchvision:", torchvision.__version__, Path(torchvision.__file__).resolve())
print("transformers:", transformers.__version__, Path(transformers.__file__).resolve())
print("peft:", peft.__version__, Path(peft.__file__).resolve())
print("CUDA available:", torch.cuda.is_available())
print("Supplemental package directory:", package_dir)
"""
    ),
    markdown("## 4. Build the signed runtime configuration"),
    code(
        """import yaml
from chembreak3.config import canonical_config, load_config

phase_counts = {"test": 8, "pilot": 40, "production": 500}
assert PHASE in phase_counts
base = load_config(PROJECT_DIR / "configs" / "config.test.yaml")
runtime = canonical_config(base)
runtime["run"]["phase"] = PHASE
runtime["run"]["dry_run"] = not LIVE
runtime["run"]["live_acknowledgement"] = LIVE
runtime["run"]["gcs_checkpoint_uri"] = GCS_CHECKPOINT_URI
runtime["run"]["output_root"] = str(storage_root / "runs")
runtime["experiment"]["task_count"] = phase_counts[PHASE]

storage = runtime["storage"]
storage.update({
    "content_root": str(content_root),
    "require_content_routing": True,
    "require_separate_mount": True,
    "minimum_free_gb": MINIMUM_FREE_GIB,
    "storage_root": str(storage_root),
    "hf_home": str(paths["HF_HOME"]),
    "hf_hub_cache": str(paths["HF_HUB_CACHE"]),
    "hf_modules_cache": str(paths["HF_MODULES_CACHE"]),
    "xdg_cache_home": str(paths["XDG_CACHE_HOME"]),
    "torch_home": str(paths["TORCH_HOME"]),
    "torchinductor_cache": str(paths["TORCHINDUCTOR_CACHE_DIR"]),
    "triton_cache": str(paths["TRITON_CACHE_DIR"]),
    "cuda_cache": str(paths["CUDA_CACHE_PATH"]),
    "pip_cache": str(paths["PIP_CACHE_DIR"]),
    "python_packages": str(package_dir),
    "temp_dir": str(paths["TMPDIR"]),
    "offload_dir": str(paths["CHEMBREAK_OFFLOAD_DIR"]),
    "preflight_dir": str(storage_root / "preflight"),
})
for target in runtime["targets"]:
    target["cache_dir"] = str(paths["HF_HUB_CACHE"])
    target["offload_folder"] = str(paths["CHEMBREAK_OFFLOAD_DIR"] / target["id"])

runtime_dir = PROJECT_DIR / "configs"
runtime_path = runtime_dir / f"runtime.{PHASE}.yaml"
runtime_path.write_text(yaml.safe_dump(runtime, sort_keys=False), encoding="utf-8")

if LIVE:
    assert GOOGLE_CLOUD_PROJECT != "YOUR_GOOGLE_CLOUD_PROJECT_ID", (
        "Replace the project placeholder before live execution."
    )
    os.environ["CHEMBREAK_ENABLE_LIVE"] = "YES"
else:
    os.environ.pop("CHEMBREAK_ENABLE_LIVE", None)
os.environ["GOOGLE_CLOUD_PROJECT"] = GOOGLE_CLOUD_PROJECT
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
print(runtime_path)
print(f"phase={PHASE} live={LIVE} tasks={phase_counts[PHASE]}")
"""
    ),
    markdown(
        """## 5. Storage and service preflight

Mock mode validates the package, task selection, imports, and disk routing without contacting models. Live mode also checks the four Google Cloud roles and Hugging Face repositories. Set `load_targets=True` only for a full sequential download and GPU-load check.
"""
    ),
    code(
        """from chembreak3.preflight import run_preflight

preflight = run_preflight(runtime_path, load_targets=False)
print({
    "status": preflight["status"],
    "storage": preflight["storage"],
    "gpu": preflight["gpu"],
    "selected_subset": preflight["selected_subset"],
})
"""
    ),
    markdown(
        """## 6. Execute or resume

Progress, completed count, and ETA remain visible. Re-running with the same signed configuration resumes the same checkpoint and skips completed episodes.
"""
    ),
    code(
        """from chembreak3.runner import run

RUN_DIR = run(runtime_path)
assert RUN_DIR.is_relative_to(content_root), RUN_DIR
print(RUN_DIR)
"""
    ),
    markdown("## 7. Review the main result tables"),
    code(
        """import pandas as pd
from IPython.display import display

display(pd.read_csv(RUN_DIR / "release" / "metrics_overall.csv"))
display(pd.read_csv(RUN_DIR / "release" / "asr_by_query_budget.csv"))
display(pd.read_csv(RUN_DIR / "release" / "paired_comparisons.csv"))
failures = pd.read_csv(RUN_DIR / "release" / "failures.csv")
print(f"recorded failures: {len(failures)}")
display(failures.head(20))
"""
    ),
    markdown(
        """## 8. Before moving phases

Do not move from test to pilot, or pilot to production, until the previous phase is complete and the failure ledger has been reviewed. Configuration changes create a new signed run directory.

To reclaim system-disk space from the abandoned V4 run, first stop its kernel activity and inspect `/home/jupyter/.cache/huggingface`. Remove that legacy directory only after confirming it contains no files you still need. ChemBreak3 never writes to or deletes it.
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

destination = Path(__file__).resolve().parents[1] / "notebooks" / "chembreak3_Cloud_Notebook.ipynb"
destination.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(destination)
