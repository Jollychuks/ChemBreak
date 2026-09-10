from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "chembreak8_Cloud_Notebook.ipynb"


def markdown(value: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(value).lstrip().splitlines(True)}


def code(value: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": dedent(value).lstrip().splitlines(True),
    }


cells = [
    markdown("""
    # ChemBreak8: Four-Condition Learned MDP Evaluation

    Standalone Google Cloud Notebook Enterprise workflow for the frozen ChemHarm task bank.
    The research target is at least 90% chemically validated ASR on untouched tasks. The
    notebook reports the measured result without changing thresholds or excluding failures.

    Raw prompts and target responses remain private in the standard workflow.
    """),
    markdown("""
    ## 1. Settings and content-disk routing

    Start with `LIVE = False`. Complete development and freeze its learned policy before pilot.
    Keep the 400-task holdout untouched until the methodology is final.
    """),
    code("""
    from pathlib import Path
    import importlib
    import json
    import os
    import shutil
    import site
    import subprocess
    import sys

    PROJECT_ID = "rs-foundsecft-mghasemi"
    REPO_URL = "https://github.com/Jollychuks/ChemBreak.git"
    BRANCH = "main"
    PROJECT_SUBDIR = "chembreak8"
    PHASE = "development"  # development, pilot, holdout, or full_bank
    EXPERIMENT_REVISION = "CB8_MDP_V1"
    LIVE = False

    content_root = Path("/content").resolve()
    assert content_root.is_dir(), "/content is unavailable"
    if LIVE:
        assert PROJECT_ID.strip() and not PROJECT_ID.startswith("REPLACE_"), "Set PROJECT_ID."
        assert REPO_URL.startswith(("https://github.com/", "git@github.com:")), "Set REPO_URL."
    assert PHASE in {"development", "pilot", "holdout", "full_bank"}

    storage_root = content_root / "chembreak8_storage"
    previous_model_cache = content_root / "chembreak7_storage" / "cache" / "huggingface" / "hub"
    model_cache = previous_model_cache if previous_model_cache.is_dir() else storage_root / "cache" / "huggingface" / "hub"
    paths = {
        "HF_HOME": storage_root / "cache/huggingface", "HF_HUB_CACHE": model_cache,
        "HF_MODULES_CACHE": storage_root / "cache/huggingface/modules",
        "XDG_CACHE_HOME": storage_root / "cache/xdg", "TORCH_HOME": storage_root / "cache/torch",
        "TORCHINDUCTOR_CACHE_DIR": storage_root / "cache/torchinductor",
        "TRITON_CACHE_DIR": storage_root / "cache/triton", "CUDA_CACHE_PATH": storage_root / "cache/cuda",
        "PIP_CACHE_DIR": storage_root / "cache/pip", "TMPDIR": storage_root / "tmp",
    }
    for variable, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    os.environ["TMP"] = os.environ["TEMP"] = str(paths["TMPDIR"])
    print("Content disk:", shutil.disk_usage(content_root))
    print("ChemBreak8 storage:", storage_root)
    print("Model cache:", model_cache)
    """),
    markdown("""
    ## 2. Obtain a clean Git checkout

    The cell uses a dedicated runtime checkout and never overwrites a dirty checkout.
    """),
    code("""
    checkout_base = content_root / "chembreak8_runtime_repo"

    def git(*args, cwd=None, capture=False):
        command = ["git", *args]
        if capture:
            return subprocess.check_output(command, cwd=cwd, text=True).strip()
        subprocess.run(command, cwd=cwd, check=True)

    checkout = checkout_base
    if (checkout / ".git").is_dir() and git("status", "--porcelain", cwd=checkout, capture=True):
        suffix = 2
        while (content_root / f"chembreak8_runtime_repo_{suffix}").exists():
            suffix += 1
        checkout = content_root / f"chembreak8_runtime_repo_{suffix}"
    if not (checkout / ".git").is_dir():
        if REPO_URL.startswith(("https://github.com/", "git@github.com:")):
            git("clone", "--branch", BRANCH, "--single-branch", REPO_URL, str(checkout))
        else:
            candidates = [path for path in content_root.glob("*") if (path / PROJECT_SUBDIR / "pyproject.toml").is_file()]
            assert candidates, "Set REPO_URL or upload the complete chembreak8 folder under /content."
            checkout = candidates[0]
    if (checkout / ".git").is_dir() and not git("status", "--porcelain", cwd=checkout, capture=True):
        git("fetch", "origin", BRANCH, cwd=checkout)
        assert git("ls-tree", "-d", "--name-only", f"origin/{BRANCH}", PROJECT_SUBDIR, cwd=checkout, capture=True) == PROJECT_SUBDIR
        git("checkout", BRANCH, cwd=checkout)
        git("pull", "--ff-only", "origin", BRANCH, cwd=checkout)
    PROJECT_DIR = (checkout / PROJECT_SUBDIR).resolve()
    assert (PROJECT_DIR / "pyproject.toml").is_file(), f"Incomplete project: {PROJECT_DIR}"
    os.chdir(PROJECT_DIR)
    print("Project:", PROJECT_DIR)
    """),
    markdown("""
    ## 3. Install the non-Torch compatibility stack

    Notebook Enterprise supplies Torch, Torchvision, and CUDA. This cell preserves that stack.
    Restart the kernel if a prior ChemBreak package stack is already loaded.
    """),
    code("""
    contaminated = []
    for module_name in ("torch", "torchvision", "transformers", "peft"):
        module = sys.modules.get(module_name)
        origin = str(getattr(module, "__file__", "")) if module else ""
        if any(f"/content/chembreak{version}_storage/python_packages/" in origin for version in range(1, 8)):
            contaminated.append(f"{module_name}={origin}")
    if contaminated:
        raise RuntimeError("Restart the kernel before running ChemBreak8. " + "; ".join(contaminated))

    package_dir = storage_root / "python_packages"
    package_dir.mkdir(parents=True, exist_ok=True)
    for forbidden in ("torch", "torchvision", "torchaudio", "torchgen", "functorch"):
        if (package_dir / forbidden).exists():
            raise RuntimeError(f"Unexpected {forbidden} under {package_dir}. Use fresh ChemBreak8 storage.")
    compatibility_specs = [
        "transformers==4.40.2", "tokenizers==0.19.1", "huggingface-hub==0.23.5",
        "safetensors==0.4.5", "accelerate==0.30.1", "peft==0.10.0",
        "sentencepiece==0.2.0", "einops==0.8.1",
    ]
    marker = package_dir / "chembreak8_compatibility_stack.json"
    expected = {"specifications": compatibility_specs}
    installed = json.loads(marker.read_text()) if marker.exists() else None
    if installed != expected:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--target", str(package_dir),
            "--cache-dir", str(paths["PIP_CACHE_DIR"]), "--no-deps", "--upgrade", *compatibility_specs,
        ], check=True)
        marker.write_text(json.dumps(expected, indent=2))
    site.addsitedir(str(package_dir))
    sys.path.insert(0, str(package_dir))
    sys.path.insert(0, str(PROJECT_DIR / "src"))
    importlib.invalidate_caches()
    required = {
        "google.auth": "google-auth>=2.35,<3", "google.cloud.storage": "google-cloud-storage>=2.18,<4",
        "google.genai": "google-genai>=1.47,<2", "openai": "openai>=1.57,<3",
        "numpy": "numpy>=1.26,<3", "pandas": "pandas>=2.2,<3", "pydantic": "pydantic>=2.9,<3",
        "yaml": "PyYAML>=6.0,<7", "rdkit": "rdkit>=2024.3", "scipy": "scipy>=1.13,<2",
        "tenacity": "tenacity>=9,<10",
    }
    missing = []
    for module_name, requirement in required.items():
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing.append(requirement)
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "--target", str(package_dir), "--cache-dir", str(paths["PIP_CACHE_DIR"]), "--upgrade", *missing], check=True)
    import torch, torchvision, transformers, peft
    assert not str(Path(torch.__file__).resolve()).startswith(str(storage_root))
    print("torch:", torch.__version__, Path(torch.__file__).resolve())
    print("torchvision:", torchvision.__version__)
    print("transformers:", transformers.__version__, "peft:", peft.__version__)
    print("CUDA available:", torch.cuda.is_available())
    """),
    markdown("""
    ## 4. Create a runtime configuration outside Git

    Development has 48 exposed tasks. The new pilot has 52 unseen tasks. The final holdout has
    400 unseen tasks. Development trains the policy. Every later phase requires the frozen policy.
    """),
    code("""
    import yaml
    from chembreak8.config import load_config

    config = load_config(PROJECT_DIR / "configs" / f"config.{PHASE}.yaml")
    config.pop("_config_path", None)
    config["run"].update({
        "project_root": str(PROJECT_DIR), "task_bank_path": str(PROJECT_DIR / "data/final_task_bank.csv"),
        "output_root": str(storage_root / "runs"), "dry_run": not LIVE,
        "live_acknowledgement": LIVE, "experiment_revision": EXPERIMENT_REVISION,
    })
    for key, value in {
        "storage_root": storage_root, "hf_home": storage_root / "cache/huggingface",
        "hf_hub_cache": model_cache, "hf_modules_cache": storage_root / "cache/huggingface/modules",
        "xdg_cache_home": storage_root / "cache/xdg", "torch_home": storage_root / "cache/torch",
        "torchinductor_cache": storage_root / "cache/torchinductor", "triton_cache": storage_root / "cache/triton",
        "cuda_cache": storage_root / "cache/cuda", "pip_cache": storage_root / "cache/pip",
        "python_packages": storage_root / "python_packages", "temp_dir": storage_root / "tmp",
        "offload_dir": storage_root / "offload", "preflight_dir": storage_root / "preflight",
    }.items():
        config["storage"][key] = str(value)
    for target in config["targets"]:
        target["cache_dir"] = str(model_cache)
        target["offload_folder"] = str(storage_root / "offload" / target["id"])
    policy_dir = storage_root / "policies" / EXPERIMENT_REVISION
    training_policy_path = policy_dir / "development_policy.json"
    frozen_policy_path = policy_dir / "frozen_policy.json"
    config["policy"]["artifact_path"] = str(training_policy_path if PHASE == "development" else frozen_policy_path)
    if PHASE != "development":
        assert frozen_policy_path.is_file(), "Complete development and freeze the policy first."
    runtime_dir = storage_root / "runtime_configs"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = runtime_dir / f"CB8_{PHASE}_{'live' if LIVE else 'mock'}.yaml"
    runtime_path.write_text(yaml.safe_dump(config, sort_keys=False))
    if LIVE:
        os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
        os.environ["CHEMBREAK_ENABLE_LIVE"] = "YES"
    else:
        os.environ.pop("CHEMBREAK_ENABLE_LIVE", None)
    print("Runtime config:", runtime_path)
    print("Mode:", "LIVE" if LIVE else "MOCK")
    print("Policy:", config["policy"]["mode"], config["policy"]["artifact_path"])
    """),
    markdown("""
    ## 5. Preflight

    Live preflight tests strict schemas for all five roles and probes the three target tokenizers.
    """),
    code("""
    from chembreak8.preflight import run_preflight
    preflight = run_preflight(runtime_path, load_targets=False, probe_tokenizers=True)
    print({key: preflight[key] for key in ("status", "gpu", "selected_subset", "policy", "roles", "tokenizers")})
    """),
    markdown("""
    ## 6. Independent condition and target runner
    """),
    code("""
    import pandas as pd
    from IPython.display import display
    from chembreak8.runner import run_condition_target

    def run_one(condition, target_id):
        run_dir = Path(run_condition_target(runtime_path, condition, target_id))
        results_path = run_dir / "release" / f"episode_results_{condition}.csv"
        metrics_path = run_dir / "release" / f"metrics_{condition}.csv"
        results, metrics = pd.read_csv(results_path), pd.read_csv(metrics_path)
        display(results[results.target_id == target_id])
        display(metrics[metrics.target_id.isin([target_id, "ALL_TARGETS"])])
        print("Episode CSV:", results_path)
        print("Metrics CSV:", metrics_path)
        return run_dir
    """),
]

for number, condition, explanation in (
    (7, "C0_DIRECT", "One frozen benchmark prompt and one query."),
    (8, "C1_REPEATED_SINGLE", "The unchanged prompt, repeated for up to eight same-session queries."),
    (9, "C2_FIXED_MULTI", "A fixed seven-action sequence that never receives target feedback."),
    (10, "C3_ADAPTIVE_MDP", "The learned adaptive policy and focus of the 90% research target."),
):
    cells.append(markdown(f"## {number}. {condition}\n\n{explanation}"))
    for target in ("ChemDFM", "ChemLLM", "LlaSMol"):
        cells.append(code(f'RUN_DIR = run_one("{condition}", "{target}")'))

cells.extend([
    markdown("""
    ## 11. Recover saved responses without repeating target queries
    """),
    code("""
    from chembreak8.runner import recover_pending
    recovery = recover_pending(runtime_path)
    print(recovery)
    RUN_DIR = Path(recovery["run_dir"])
    """),
    markdown("""
    ## 12. Condition completion gates
    """),
    code("""
    from chembreak8.runner import strict_completion_gate
    from chembreak8.schema import CONDITIONS
    for condition in CONDITIONS:
        try:
            print(strict_completion_gate(runtime_path, condition))
        except RuntimeError as error:
            print(condition, "INCOMPLETE:", error)
    """),
    markdown("""
    ## 13. Freeze the C3 policy after completed development
    """),
    code("""
    from chembreak8.policy import freeze_policy
    if PHASE == "development":
        strict_completion_gate(runtime_path, "C3_ADAPTIVE_MDP")
        print(freeze_policy(training_policy_path, frozen_policy_path))
    else:
        print("No freeze performed. This phase uses the frozen policy.")
    """),
    markdown("## 14. Consolidated redacted results"),
    code("""
    display(pd.read_csv(Path(RUN_DIR) / "release/adaptive_mdp_metrics.csv"))
    display(pd.read_csv(Path(RUN_DIR) / "release/run_coverage.csv"))
    display(pd.read_csv(Path(RUN_DIR) / "release/asr_by_query_budget.csv"))
    """),
    markdown("## 15. Download CSV files and the redacted archive"),
    code("""
    from IPython.display import FileLink, display
    release_dir = Path(RUN_DIR) / "release"
    for condition in ("C0_DIRECT", "C1_REPEATED_SINGLE", "C2_FIXED_MULTI", "C3_ADAPTIVE_MDP"):
        for name in (f"episode_results_{condition}.csv", f"metrics_{condition}.csv", f"asr_by_budget_{condition}.csv"):
            path = release_dir / name
            if path.exists():
                display(FileLink(str(path)))
    downloads = storage_root / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    archive_path = Path(shutil.make_archive(str(downloads / f"{Path(RUN_DIR).name}_release"), "zip", root_dir=Path(RUN_DIR), base_dir="release"))
    print("Release archive:", archive_path)
    display(FileLink(str(archive_path)))
    """),
    markdown("""
    ## 16. Optional private review

    Private files may contain harmful task text and model output. Review is disabled by default.
    """),
    code("""
    ENABLE_PRIVATE_REVIEW = False
    if ENABLE_PRIVATE_REVIEW:
        display(pd.read_csv(Path(RUN_DIR) / "private/successful_attacks_private.csv"))
    else:
        print("Private review is disabled.")
    """),
])

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(OUTPUT)
