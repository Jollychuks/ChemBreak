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
        """# ChemBreak6 on Google Cloud Notebook Enterprise

ChemBreak6 runs the frozen ChemHarm task bank with a shared signed checkpoint and one independent section for each attack condition. Run Sections 1 through 5 in order. Then run any condition section independently. Completed episodes are skipped when a cell is rerun.

All model caches, packages, checkpoints, temporary files, and results are routed to `/content`. Raw attack prompts and target responses stay private by default.
"""
    ),
    code(
        """# EDIT THIS CELL
GOOGLE_CLOUD_PROJECT = "YOUR_GOOGLE_CLOUD_PROJECT_ID"
PHASE = "test"  # test, pilot, or production
LIVE = False  # complete a mock test before changing this to True
SHOW_PRIVATE_OUTPUTS = False
GCS_CHECKPOINT_URI = None  # example: gs://private-bucket/chembreak/chembreak6
REPO_URL = "https://github.com/Jollychuks/ChemBreak.git"
BRANCH = "main"
PROJECT_SUBDIR = "chembreak6"

CONTENT_ROOT = "/content"
STORAGE_ROOT = "/content/chembreak6_storage"
MINIMUM_FREE_GIB = 100
"""
    ),
    markdown("## 1. Route heavyweight writes to the content disk"),
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

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
usage = shutil.disk_usage(content_root)
free_gib = usage.free / 1024**3
assert free_gib >= MINIMUM_FREE_GIB, (
    f"Only {free_gib:.1f} GiB is free on {content_root}; {MINIMUM_FREE_GIB} GiB is required."
)
subprocess.run(["df", "-h", "/", str(content_root)], check=True)
print(f"CB6 storage root: {storage_root}")
print(f"Free on content disk: {free_gib:.1f} GiB")
"""
    ),
    markdown(
        """## 2. Create an isolated runtime checkout

Upload the complete `chembreak6` folder to the root of the GitHub repository before running this cell. Runtime configuration files are written outside Git, so notebook execution cannot make this checkout dirty. If an unexpected local change exists, the cell creates a second checkout at the remote commit instead of resetting or deleting anything.
"""
    ),
    code(
        """base_checkout = content_root / "chembreak6_runtime_repo"

def clone_checkout(destination):
    subprocess.run(
        ["git", "clone", "--branch", BRANCH, "--single-branch", REPO_URL, str(destination)],
        check=True,
    )

if not base_checkout.exists():
    clone_checkout(base_checkout)
    checkout = base_checkout
else:
    assert (base_checkout / ".git").is_dir(), f"Not a Git checkout: {base_checkout}"
    subprocess.run(["git", "-C", str(base_checkout), "fetch", "origin", BRANCH], check=True)
    changed = subprocess.run(
        ["git", "-C", str(base_checkout), "status", "--porcelain"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if not changed:
        subprocess.run(
            ["git", "-C", str(base_checkout), "pull", "--ff-only", "origin", BRANCH],
            check=True,
        )
        checkout = base_checkout
    else:
        remote_commit = subprocess.run(
            ["git", "-C", str(base_checkout), "rev-parse", f"origin/{BRANCH}"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        checkout = content_root / f"chembreak6_runtime_repo_{remote_commit[:12]}"
        if not checkout.exists():
            clone_checkout(checkout)
        print("The original runtime checkout has local changes and was left untouched.")
        print("Using isolated checkout:", checkout)

subprocess.run(["git", "-C", str(checkout), "fetch", "origin", BRANCH], check=True)
remote_tree = subprocess.run(
    ["git", "-C", str(checkout), "ls-tree", "-d", "--name-only", f"origin/{BRANCH}"],
    check=True, capture_output=True, text=True,
).stdout.splitlines()
assert PROJECT_SUBDIR in remote_tree, (
    f"origin/{BRANCH} does not contain {PROJECT_SUBDIR}. Upload and commit the folder first."
)

PROJECT_DIR = (checkout / PROJECT_SUBDIR).resolve()
assert PROJECT_DIR.is_relative_to(content_root), PROJECT_DIR
assert (PROJECT_DIR / "pyproject.toml").is_file(), f"Project not found: {PROJECT_DIR}"
commit = subprocess.run(
    ["git", "-C", str(checkout), "rev-parse", "HEAD"],
    check=True, capture_output=True, text=True,
).stdout.strip()
os.chdir(PROJECT_DIR)
print("Project:", PROJECT_DIR)
print("Git commit:", commit)
"""
    ),
    markdown(
        """## 3. Install the non-Torch compatibility stack

Restart the kernel first if an earlier ChemBreak notebook installed or imported packages. This cell preserves the Notebook Enterprise Torch, Torchvision, and CUDA stack. It never installs those packages.
"""
    ),
    code(
        """import importlib
import json
import site

contaminated_modules = []
for module_name in ("torch", "torchvision", "transformers", "peft"):
    module = sys.modules.get(module_name)
    origin = str(getattr(module, "__file__", "")) if module else ""
    if any(f"/content/chembreak{version}_storage/" in origin for version in (1, 2, 3, 4, 5)):
        contaminated_modules.append(f"{module_name}={origin}")
if contaminated_modules:
    raise RuntimeError(
        "This kernel contains packages from an earlier ChemBreak version. Restart the kernel, "
        "open the ChemBreak6 notebook, and run from the first cell. " + "; ".join(contaminated_modules)
    )

package_dir = storage_root / "python_packages"
package_dir.mkdir(parents=True, exist_ok=True)
for forbidden in ("torch", "torchvision", "torchaudio", "torchgen", "functorch"):
    if (package_dir / forbidden).exists():
        raise RuntimeError(
            f"Unexpected {forbidden} package found in {package_dir}. Use a fresh "
            "chembreak6_storage directory before continuing."
        )

compatibility_specs = [
    "transformers==4.40.2",
    "tokenizers==0.19.1",
    "huggingface-hub==0.23.5",
    "safetensors==0.4.5",
    "accelerate==0.30.1",
    "peft==0.10.0",
    "sentencepiece==0.2.0",
    "einops==0.8.1",
]
compatibility_marker = package_dir / "chembreak6_compatibility_stack.json"
expected_marker = {
    "python": f"{sys.version_info.major}.{sys.version_info.minor}",
    "specifications": compatibility_specs,
}
installed_marker = None
if compatibility_marker.exists():
    try:
        installed_marker = json.loads(compatibility_marker.read_text(encoding="utf-8"))
    except Exception:
        installed_marker = None

if installed_marker != expected_marker:
    print("Installing the ChemBreak6 non-Torch compatibility stack on /content.")
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "--target", str(package_dir),
            "--cache-dir", str(paths["PIP_CACHE_DIR"]), "--no-deps", "--upgrade",
            *compatibility_specs,
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

required = {
    "google.auth": "google-auth>=2.35,<3",
    "google.cloud.storage": "google-cloud-storage>=2.18,<4",
    "google.genai": "google-genai>=1.47,<2",
    "openai": "openai>=1.57,<3",
    "numpy": "numpy>=1.26,<3",
    "pandas": "pandas>=2.2,<3",
    "pydantic": "pydantic>=2.9,<3",
    "yaml": "PyYAML>=6.0,<7",
    "rdkit": "rdkit>=2024.3",
    "scipy": "scipy>=1.13,<2",
    "tenacity": "tenacity>=9,<10",
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
            sys.executable, "-m", "pip", "install", "--target", str(package_dir),
            "--cache-dir", str(paths["PIP_CACHE_DIR"]), "--upgrade", *missing_specs,
        ],
        check=True,
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
import accelerate
import tokenizers
import huggingface_hub
import einops
import chembreak6

torch_origin = str(Path(torch.__file__).resolve())
assert "/content/chembreak" not in torch_origin or "_storage/" not in torch_origin, (
    f"Torch must come from the Notebook Enterprise image, not ChemBreak storage: {torch_origin}"
)
expected_versions = {
    "transformers": "4.40.2", "peft": "0.10.0", "accelerate": "0.30.1",
    "tokenizers": "0.19.1", "huggingface_hub": "0.23.5", "einops": "0.8.1",
}
active_versions = {
    "transformers": transformers.__version__, "peft": peft.__version__,
    "accelerate": accelerate.__version__, "tokenizers": tokenizers.__version__,
    "huggingface_hub": huggingface_hub.__version__, "einops": einops.__version__,
}
assert active_versions == expected_versions, (
    f"The compatibility stack is not active: {active_versions}. Restart the kernel and rerun."
)
for module in (transformers, peft, accelerate, tokenizers, huggingface_hub, einops):
    assert str(Path(module.__file__).resolve()).startswith(str(package_dir)), (
        f"Compatibility package loaded outside {package_dir}: {module.__file__}"
    )
assert chembreak6.__version__ == "6.0.0", chembreak6.__version__
compatibility_marker.write_text(json.dumps(expected_marker, indent=2), encoding="utf-8")
print("torch:", torch.__version__, torch_origin)
print("torchvision:", torchvision.__version__, Path(torchvision.__file__).resolve())
print("ChemBreak6 compatibility stack:", active_versions)
print("CUDA available:", torch.cuda.is_available())
print("Supplemental package directory:", package_dir)
"""
    ),
    markdown("## 4. Build one signed runtime configuration"),
    code(
        """import yaml
from chembreak6.config import canonical_config, load_config

phase_counts = {"test": 8, "pilot": 40, "production": 500}
assert PHASE in phase_counts
base = load_config(PROJECT_DIR / "configs" / "config.test.yaml")
runtime = canonical_config(base)
runtime["run"]["project_root"] = str(PROJECT_DIR)
runtime["run"]["task_bank_path"] = str(PROJECT_DIR / "data" / "final_task_bank.csv")
runtime["run"]["phase"] = PHASE
runtime["run"]["dry_run"] = not LIVE
runtime["run"]["live_acknowledgement"] = LIVE
runtime["run"]["gcs_checkpoint_uri"] = GCS_CHECKPOINT_URI
runtime["run"]["output_root"] = str(storage_root / "runs")
runtime["experiment"]["task_count"] = phase_counts[PHASE]

storage = runtime["storage"]
storage.update({
    "content_root": str(content_root), "require_content_routing": True,
    "require_separate_mount": True, "minimum_free_gb": MINIMUM_FREE_GIB,
    "storage_root": str(storage_root), "hf_home": str(paths["HF_HOME"]),
    "hf_hub_cache": str(paths["HF_HUB_CACHE"]),
    "hf_modules_cache": str(paths["HF_MODULES_CACHE"]),
    "xdg_cache_home": str(paths["XDG_CACHE_HOME"]),
    "torch_home": str(paths["TORCH_HOME"]),
    "torchinductor_cache": str(paths["TORCHINDUCTOR_CACHE_DIR"]),
    "triton_cache": str(paths["TRITON_CACHE_DIR"]),
    "cuda_cache": str(paths["CUDA_CACHE_PATH"]),
    "pip_cache": str(paths["PIP_CACHE_DIR"]), "python_packages": str(package_dir),
    "temp_dir": str(paths["TMPDIR"]), "offload_dir": str(paths["CHEMBREAK_OFFLOAD_DIR"]),
    "preflight_dir": str(storage_root / "preflight"),
})
for target in runtime["targets"]:
    target["cache_dir"] = str(paths["HF_HUB_CACHE"])
    target["offload_folder"] = str(paths["CHEMBREAK_OFFLOAD_DIR"] / target["id"])

runtime_dir = storage_root / "runtime"
runtime_dir.mkdir(parents=True, exist_ok=True)
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
print("Condition budgets:", runtime["experiment"]["condition_query_budgets"])
print("LIVE JAILBREAK EVALUATION" if LIVE else "MOCK VALIDATION ONLY")
"""
    ),
    markdown(
        """## 5. Preflight

Live preflight checks the five model roles with their production response protocols, then probes every tokenizer before any target weights are loaded. The GPT-OSS safety judge uses client-validated compact JSON. Gemini roles use response schemas.
"""
    ),
    code(
        """from chembreak6.preflight import run_preflight

preflight = run_preflight(runtime_path, load_targets=False, probe_tokenizers=True)
print("Preflight status:", preflight["status"])
print("GPU:", preflight["gpu"])
print("Selected subset:", preflight["selected_subset"])
print("Role checks:")
for role in preflight["roles"]:
    print(role)
print("Tokenizer checks:")
for tokenizer in preflight["tokenizers"]:
    print(tokenizer)
"""
    ),
    markdown(
        """## Condition execution

Each condition cell uses the same signed run directory, frozen task selection, and SQLite checkpoint. Re-running a cell does not repeat completed episodes. Separate cells do not increase API calls by themselves. They can increase elapsed GPU time because each cell loads the three target models sequentially again.
"""
    ),
    code(
        """import pandas as pd
from IPython.display import display
from chembreak6.runner import run_condition

def show_condition_results(run_dir, condition):
    path = run_dir / "release" / "by_condition" / f"{condition}_episode_results.csv"
    rows = pd.read_csv(path)
    display(rows[[
        "assignment_id", "target_id", "condition", "status", "queries_used",
        "final_response_class", "chemistry_validation", "success_label", "terminal_reason",
        "post_feedback_success",
    ]])
    print("CSV:", path)
"""
    ),
    markdown(
        """## 6A. C0_DIRECT

True direct control. It sends the unmodified benchmark prompt once per task and target. Test phase maximum: 24 target queries.
"""
    ),
    code(
        """RUN_DIR = run_condition(runtime_path, "C0_DIRECT")
show_condition_results(RUN_DIR, "C0_DIRECT")
"""
    ),
    markdown(
        """## 6B. C1_REPEATED_SINGLE

Five independently generated single-turn attempts. Target conversation history is reset for every attempt. Test phase maximum: 120 target queries.
"""
    ),
    code(
        """RUN_DIR = run_condition(runtime_path, "C1_REPEATED_SINGLE")
show_condition_results(RUN_DIR, "C1_REPEATED_SINGLE")
"""
    ),
    markdown(
        """## 6C. C2_FIXED_MULTI

One five-turn conversation planned in advance. It sees target conversation history but does not change its plan based on responses. Test phase maximum: 120 target queries.
"""
    ),
    code(
        """RUN_DIR = run_condition(runtime_path, "C2_FIXED_MULTI")
show_condition_results(RUN_DIR, "C2_FIXED_MULTI")
"""
    ),
    markdown(
        """## 6D. C3_ADAPTIVE_MDP

Maximum five-turn adaptive condition. From turn two onward, the planner sees raw recent conversation only. It never receives judge scores, labels, rationales, or rewards. A turn-one success is reported as a C3 success but not as a post-feedback adaptive success.
"""
    ),
    code(
        """RUN_DIR = run_condition(runtime_path, "C3_ADAPTIVE_MDP")
show_condition_results(RUN_DIR, "C3_ADAPTIVE_MDP")
"""
    ),
    markdown(
        """## 7. Recover saved responses awaiting judgment

Run this after a temporary structured-output failure. It retries only the missing judgments and does not issue another target-model query. Episodes that still have budget become ready for their condition cell to continue.
"""
    ),
    code(
        """from chembreak6.runner import recover_pending_judgments

recovery = recover_pending_judgments(runtime_path)
print(recovery)
print("Rerun the relevant condition cell if recovered episodes are ready to continue.")
"""
    ),
    markdown("## 8. Review partial results at any time"),
    code(
        """from chembreak6.runner import finalize_run

status = finalize_run(runtime_path, require_complete=False)
RUN_DIR = Path(status["run_dir"])
print(status)
episode_results = pd.read_csv(RUN_DIR / "release" / "episode_results.csv")
success_table = pd.read_csv(RUN_DIR / "release" / "success_by_target_condition.csv")
coverage = pd.read_csv(RUN_DIR / "release" / "run_coverage.csv")
adaptive = pd.read_csv(RUN_DIR / "release" / "adaptive_mdp_metrics.csv")
display(success_table)
display(coverage)
display(adaptive)
display(episode_results[[
    "assignment_id", "target_id", "condition", "status", "queries_used",
    "final_response_class", "chemistry_validation", "success_label", "terminal_reason",
    "turn_one_success", "post_feedback_success",
]])
print("Main result CSV:", RUN_DIR / "release" / "episode_results.csv")
print("Condition CSV folder:", RUN_DIR / "release" / "by_condition")
"""
    ),
    markdown(
        """## 9. Completeness gate

Run this before moving from test to pilot or from pilot to production. It refuses to certify a phase with missing, pending, failed, or unavailable episodes.
"""
    ),
    code(
        """complete_status = finalize_run(runtime_path, require_complete=True)
print("CHEMBREAK6 PHASE COMPLETE")
print(complete_status)
"""
    ),
    markdown("## 10. Create a downloadable release archive"),
    code(
        """import shutil
from IPython.display import FileLink, display

download_dir = storage_root / "downloads"
download_dir.mkdir(parents=True, exist_ok=True)
archive_base = download_dir / f"{RUN_DIR.name}_release"
archive_path = Path(shutil.make_archive(str(archive_base), "zip", RUN_DIR / "release"))
print("Release archive:", archive_path)
display(FileLink(str(archive_path)))
"""
    ),
    markdown("## 11. Optional private review"),
    code(
        """if SHOW_PRIVATE_OUTPUTS:
    private_successes = pd.read_csv(RUN_DIR / "private" / "successful_attacks_private.csv")
    print("PRIVATE CONTROLLED-RESEARCH OUTPUT. Do not publish raw prompts or responses.")
    display(private_successes)
else:
    print("Private raw successful attacks are hidden by default.")
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

destination = Path(__file__).resolve().parents[1] / "notebooks" / "chembreak6_Cloud_Notebook.ipynb"
destination.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(destination)
