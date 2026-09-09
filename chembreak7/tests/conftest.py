from __future__ import annotations

from pathlib import Path

import yaml

from chembreak7.config import canonical_config, load_config

ROOT = Path(__file__).resolve().parents[1]


def make_config(tmp_path: Path, phase: str = "development") -> Path:
    config = canonical_config(load_config(ROOT / "configs" / f"config.{phase}.yaml"))
    content = tmp_path / "content"
    storage = content / "chembreak7_storage"
    content.mkdir(parents=True, exist_ok=True)
    config["run"].update({
        "project_root": str(ROOT),
        "task_bank_path": str(ROOT / "data" / "final_task_bank.csv"),
        "output_root": str(storage / "runs"),
        "dry_run": True,
        "live_acknowledgement": False,
    })
    config["storage"].update({
        "content_root": str(content), "storage_root": str(storage),
        "require_separate_mount": False, "require_content_routing": False,
        "minimum_free_gb": 0,
        "hf_home": str(storage / "cache/huggingface"),
        "hf_hub_cache": str(storage / "cache/huggingface/hub"),
        "hf_modules_cache": str(storage / "cache/huggingface/modules"),
        "xdg_cache_home": str(storage / "cache/xdg"),
        "torch_home": str(storage / "cache/torch"),
        "torchinductor_cache": str(storage / "cache/torchinductor"),
        "triton_cache": str(storage / "cache/triton"),
        "cuda_cache": str(storage / "cache/cuda"),
        "pip_cache": str(storage / "cache/pip"),
        "python_packages": str(storage / "python_packages"),
        "temp_dir": str(storage / "tmp"),
        "offload_dir": str(storage / "offload"),
        "preflight_dir": str(storage / "preflight"),
    })
    for target in config["targets"]:
        target["cache_dir"] = str(storage / "cache/huggingface/hub")
        target["offload_folder"] = str(storage / "offload" / target["id"])
    path = tmp_path / f"config.{phase}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path

