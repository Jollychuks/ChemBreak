from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "chembreak7_Adaptive_MDP_Cloud_Notebook.ipynb"


def markdown(value: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(value).lstrip().splitlines(True)}


def code(value: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": dedent(value).lstrip().splitlines(True),
    }


cells = [
    markdown("""
    # ChemBreak7: Adaptive MDP only

    Standalone Google Cloud Notebook Enterprise workflow for the frozen ChemHarm task bank.

    ChemBreak7 reports three different quantities:

    - Overall verified ASR.
    - Bootstrap ASR from the original prompt at turn 1.
    - Conditional adaptive ASR from verified successes at turns 2 through 5 among bootstrap failures.

    Raw prompts and target responses are private and are never printed by the standard workflow.
    """),
    markdown("""
    ## 1. User settings and content-disk routing

    Start with `LIVE = False`. For live execution, use the development phase first. The holdout phase must remain untouched until the policy is frozen.
    """),
    code("""
    from pathlib import Path
    import json
    import os
    import shutil
    import site
    import subprocess
    import sys

    PROJECT_ID = "REPLACE_WITH_YOUR_PROJECT_ID"
    REPO_URL = "REPLACE_WITH_YOUR_GITHUB_REPOSITORY_URL"
    BRANCH = "main"
    PROJECT_SUBDIR = "chembreak7"
    PHASE = "development"  # development, pilot, holdout, or full_bank
    LIVE = False

    content_root = Path("/content").resolve()
    assert content_root.is_dir(), "/content is unavailable"
    if LIVE:
        assert PROJECT_ID != "REPLACE_WITH_YOUR_PROJECT_ID", "Set PROJECT_ID before a live run."
        assert REPO_URL != "REPLACE_WITH_YOUR_GITHUB_REPOSITORY_URL", "Set REPO_URL."
    assert PHASE in {"development", "pilot", "holdout", "full_bank"}

    storage_root = content_root / "chembreak7_storage"
    paths = {
        "HF_HOME": storage_root / "cache" / "huggingface",
        "HF_HUB_CACHE": storage_root / "cache" / "huggingface" / "hub",
        "HF_MODULES_CACHE": storage_root / "cache" / "huggingface" / "modules",
        "XDG_CACHE_HOME": storage_root / "cache" / "xdg",
        "TORCH_HOME": storage_root / "cache" / "torch",
        "TORCHINDUCTOR_CACHE_DIR": storage_root / "cache" / "torchinductor",
        "TRITON_CACHE_DIR": storage_root / "cache" / "triton",
        "CUDA_CACHE_PATH": storage_root / "cache" / "cuda",
        "PIP_CACHE_DIR": storage_root / "cache" / "pip",
        "TMPDIR": storage_root / "tmp",
    }
    for variable, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    os.environ["TMP"] = str(paths["TMPDIR"])
    os.environ["TEMP"] = str(paths["TMPDIR"])
    print("Content disk:", shutil.disk_usage(content_root))
    print("ChemBreak7 storage:", storage_root)
    """),
    markdown("""
    ## 2. Obtain a clean Git checkout

    This cell uses a dedicated runtime checkout. It never resets or overwrites a dirty checkout.
    """),
    code("""
    checkout_base = content_root / "chembreak7_runtime_repo"

    def git(*args, cwd=None, capture=False):
        command = ["git", *args]
        if capture:
            return subprocess.check_output(command, cwd=cwd, text=True).strip()
        subprocess.run(command, cwd=cwd, check=True)

    checkout = checkout_base
    if (checkout / ".git").is_dir():
        dirty = bool(git("status", "--porcelain", cwd=checkout, capture=True))
        if dirty:
            suffix = 2
            while (content_root / f"chembreak7_runtime_repo_{suffix}").exists():
                suffix += 1
            checkout = content_root / f"chembreak7_runtime_repo_{suffix}"
            print("The prior runtime checkout has local changes. Using:", checkout)

    if not (checkout / ".git").is_dir():
        if REPO_URL == "REPLACE_WITH_YOUR_GITHUB_REPOSITORY_URL":
            candidates = [path for path in content_root.glob("*") if (path / PROJECT_SUBDIR / "pyproject.toml").is_file()]
            assert candidates, "Set REPO_URL or place a complete chembreak7 folder in an existing /content checkout."
            checkout = candidates[0]
        else:
            git("clone", "--branch", BRANCH, "--single-branch", REPO_URL, str(checkout))

    if (checkout / ".git").is_dir() and not git("status", "--porcelain", cwd=checkout, capture=True):
        git("fetch", "origin", BRANCH, cwd=checkout)
        remote_folder = git(
            "ls-tree", "-d", "--name-only", f"origin/{BRANCH}", PROJECT_SUBDIR,
            cwd=checkout, capture=True,
        )
        assert remote_folder == PROJECT_SUBDIR, f"origin/{BRANCH} has no {PROJECT_SUBDIR} folder"
        git("checkout", BRANCH, cwd=checkout)
        git("pull", "--ff-only", "origin", BRANCH, cwd=checkout)

    PROJECT_DIR = (checkout / PROJECT_SUBDIR).resolve()
    assert (PROJECT_DIR / "pyproject.toml").is_file(), f"Incomplete project: {PROJECT_DIR}"
    os.chdir(PROJECT_DIR)
    print("Project:", PROJECT_DIR)
    """),
    markdown("""
    ## 3. Install the non-Torch compatibility stack

    Google Cloud's matched Torch, Torchvision, and CUDA stack is preserved. The cell imports `json` before writing its compatibility marker and checks for packages loaded by ChemBreak1 through ChemBreak6.
    """),
    code("""
    import importlib

    contaminated = []
    for module_name in ("torch", "torchvision", "transformers", "peft"):
        module = sys.modules.get(module_name)
        origin = str(getattr(module, "__file__", "")) if module else ""
        if any(f"/content/chembreak{version}_storage/" in origin for version in range(1, 7)):
            contaminated.append(f"{module_name}={origin}")
    if contaminated:
        raise RuntimeError("Restart the kernel before running ChemBreak7. " + "; ".join(contaminated))

    package_dir = storage_root / "python_packages"
    package_dir.mkdir(parents=True, exist_ok=True)
    for forbidden in ("torch", "torchvision", "torchaudio", "torchgen", "functorch"):
        if (package_dir / forbidden).exists():
            raise RuntimeError(f"Unexpected {forbidden} under {package_dir}. Use a fresh ChemBreak7 storage folder.")

    compatibility_specs = [
        "transformers==4.40.2", "tokenizers==0.19.1", "huggingface-hub==0.23.5",
        "safetensors==0.4.5", "accelerate==0.30.1", "peft==0.10.0",
        "sentencepiece==0.2.0", "einops==0.8.1",
    ]
    marker = package_dir / "chembreak7_compatibility_stack.json"
    expected = {"specifications": compatibility_specs}
    installed = None
    if marker.exists():
        try:
            installed = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            installed = None
    if installed != expected:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--target", str(package_dir),
            "--cache-dir", str(paths["PIP_CACHE_DIR"]), "--no-deps", "--upgrade",
            *compatibility_specs,
        ], check=True)
        marker.write_text(json.dumps(expected, indent=2), encoding="utf-8")

    site.addsitedir(str(package_dir))
    sys.path.insert(0, str(package_dir))
    sys.path.insert(0, str(PROJECT_DIR / "src"))
    os.environ["PYTHONPATH"] = os.pathsep.join([
        str(PROJECT_DIR / "src"), str(package_dir), os.environ.get("PYTHONPATH", "")
    ])
    importlib.invalidate_caches()

    required = {
        "google.auth": "google-auth>=2.35,<3",
        "google.cloud.storage": "google-cloud-storage>=2.18,<4",
        "google.genai": "google-genai>=1.47,<2",
        "openai": "openai>=1.57,<3",
        "numpy": "numpy>=1.26,<3", "pandas": "pandas>=2.2,<3",
        "pydantic": "pydantic>=2.9,<3", "yaml": "PyYAML>=6.0,<7",
        "rdkit": "rdkit>=2024.3", "scipy": "scipy>=1.13,<2", "tenacity": "tenacity>=9,<10",
    }
    missing = []
    for module_name, requirement in required.items():
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing.append(requirement)
    if missing:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--target", str(package_dir),
            "--cache-dir", str(paths["PIP_CACHE_DIR"]), "--upgrade", *missing,
        ], check=True)
    importlib.invalidate_caches()

    import torch, torchvision, transformers, peft, accelerate, tokenizers, huggingface_hub, einops
    torch_origin = str(Path(torch.__file__).resolve())
    assert not torch_origin.startswith(str(storage_root)), torch_origin
    expected_versions = {
        "transformers": "4.40.2", "peft": "0.10.0", "accelerate": "0.30.1",
        "tokenizers": "0.19.1", "huggingface_hub": "0.23.5", "einops": "0.8.1",
    }
    active_versions = {
        "transformers": transformers.__version__, "peft": peft.__version__,
        "accelerate": accelerate.__version__, "tokenizers": tokenizers.__version__,
        "huggingface_hub": huggingface_hub.__version__, "einops": einops.__version__,
    }
    assert active_versions == expected_versions, f"Restart the kernel and rerun: {active_versions}"
    print("torch:", torch.__version__, torch_origin)
    print("torchvision:", torchvision.__version__, Path(torchvision.__file__).resolve())
    print("ChemBreak7 compatibility stack:", active_versions)
    print("CUDA available:", torch.cuda.is_available())
    """),
    markdown("""
    ## 4. Create the runtime configuration outside Git

    `development` reuses the eight tasks already seen during ChemBreak6 and is the only phase intended for policy iteration. `pilot` and `holdout` are disjoint. Do not inspect holdout prompts before the policy is frozen.
    """),
    code("""
    import yaml
    from chembreak7.config import load_config

    base_path = PROJECT_DIR / "configs" / f"config.{PHASE}.yaml"
    config = load_config(base_path)
    config.pop("_config_path", None)
    config.pop("_base_", None)
    config["run"].update({
        "project_root": str(PROJECT_DIR), "task_bank_path": str(PROJECT_DIR / "data" / "final_task_bank.csv"),
        "output_root": str(storage_root / "runs"), "dry_run": not LIVE,
        "live_acknowledgement": LIVE,
    })
    runtime_dir = storage_root / "runtime_configs"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = runtime_dir / f"CB7_{PHASE}_{'live' if LIVE else 'mock'}.yaml"
    runtime_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    if LIVE:
        os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
        os.environ["CHEMBREAK_ENABLE_LIVE"] = "YES"
    else:
        os.environ.pop("CHEMBREAK_ENABLE_LIVE", None)
    print("Runtime config:", runtime_path)
    print("Mode:", "LIVE" if LIVE else "MOCK")
    print("Verification profile:", config["experiment"]["verification_mode"])
    print("Maximum target calls for this phase:", config["experiment"]["task_count"] * 3 * 5)
    """),
    markdown("""
    ## 5. Preflight

    Live preflight uses strict response schemas for all five roles and probes the three target tokenizers without loading target weights.
    """),
    code("""
    from chembreak7.preflight import run_preflight

    preflight = run_preflight(runtime_path, load_targets=False, probe_tokenizers=True)
    print({
        "status": preflight["status"], "gpu": preflight["gpu"],
        "selected_subset": preflight["selected_subset"], "roles": preflight["roles"],
        "tokenizers": preflight["tokenizers"],
    })
    """),
    markdown("""
    ## 6. Result helper

    This displays redacted episode decisions and the adaptive metrics. It never displays raw test prompts or target responses.
    """),
    code("""
    import pandas as pd
    from IPython.display import display

    def show_target_results(run_dir, target_id):
        run_dir = Path(run_dir)
        results_path = run_dir / "release" / "episode_results.csv"
        metrics_path = run_dir / "release" / "adaptive_mdp_metrics.csv"
        if results_path.exists():
            results = pd.read_csv(results_path)
            display(results[results.target_id == target_id])
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path)
            display(metrics[metrics.target_id.isin([target_id, "ALL_TARGETS"])])
        print("Episode CSV:", results_path)
        print("Adaptive metrics CSV:", metrics_path)
    """),
    markdown("""
    ## 7A. Run adaptive MDP against ChemDFM

    Run independently. The cell loads ChemDFM once, resumes its checkpoint, exports CSV files, and unloads the model.
    """),
    code("""
    from chembreak7.runner import run_target

    RUN_DIR = run_target(runtime_path, "ChemDFM")
    show_target_results(RUN_DIR, "ChemDFM")
    """),
    markdown("""
    ## 7B. Run adaptive MDP against ChemLLM
    """),
    code("""
    RUN_DIR = run_target(runtime_path, "ChemLLM")
    show_target_results(RUN_DIR, "ChemLLM")
    """),
    markdown("""
    ## 7C. Run adaptive MDP against LlaSMol
    """),
    code("""
    RUN_DIR = run_target(runtime_path, "LlaSMol")
    show_target_results(RUN_DIR, "LlaSMol")
    """),
    markdown("""
    ## 8. Recover saved responses without repeating target queries

    Run after an observer or verifier formatting failure. It resumes only the missing evaluation stage. Then rerun the relevant target cell if the episode still needs another adaptive query.
    """),
    code("""
    from chembreak7.runner import recover_pending

    recovery = recover_pending(runtime_path)
    print(recovery)
    RUN_DIR = Path(recovery["run_dir"])
    """),
    markdown("""
    ## 9. Consolidated redacted results
    """),
    code("""
    results_path = Path(RUN_DIR) / "release" / "episode_results.csv"
    metrics_path = Path(RUN_DIR) / "release" / "adaptive_mdp_metrics.csv"
    coverage_path = Path(RUN_DIR) / "release" / "run_coverage.csv"
    display(pd.read_csv(metrics_path))
    display(pd.read_csv(coverage_path))
    print("Episode results:", results_path)
    print("Action metrics:", Path(RUN_DIR) / "release" / "action_metrics.csv")
    print("State transitions:", Path(RUN_DIR) / "release" / "state_action_transitions.csv")
    """),
    markdown("""
    ## 10. Strict completion gate

    This refuses to certify ASR if any expected target episode is incomplete.
    """),
    code("""
    from chembreak7.runner import strict_completion_gate

    completion = strict_completion_gate(runtime_path)
    print(completion)
    RUN_DIR = Path(completion["run_dir"])
    """),
    markdown("""
    ## 11. Create a downloadable redacted result archive
    """),
    code("""
    import shutil
    from IPython.display import FileLink, display

    export_root = storage_root / "downloads"
    export_root.mkdir(parents=True, exist_ok=True)
    archive_base = export_root / f"{Path(RUN_DIR).name}_release"
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=Path(RUN_DIR), base_dir="release"))
    print("Release archive:", archive_path)
    display(FileLink(str(archive_path)))
    """),
    markdown("""
    ## 12. Optional private review

    Disabled by default. Private files may contain harmful task text and raw model output.
    """),
    code("""
    ENABLE_PRIVATE_REVIEW = False
    if ENABLE_PRIVATE_REVIEW:
        private_path = Path(RUN_DIR) / "private" / "successful_attacks_private.csv"
        display(pd.read_csv(private_path))
    else:
        print("Private review is disabled.")
    """),
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
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(OUTPUT)
