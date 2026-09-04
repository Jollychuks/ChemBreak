from pathlib import Path

from chembreak2.config import load_config
from chembreak2.storage import ENVIRONMENT_PATHS, is_within


ROOT = Path(__file__).resolve().parents[1]


def test_default_large_paths_are_under_content():
    config = load_config(ROOT / "configs" / "config.test.yaml")
    content = config["storage"]["content_root"]
    paths = [
        config["run"]["output_root"],
        config["storage"]["storage_root"],
        config["storage"]["python_packages"],
        config["storage"]["offload_dir"],
        config["storage"]["preflight_dir"],
        *(config["storage"][key] for key in ENVIRONMENT_PATHS),
    ]
    for target in config["targets"]:
        paths.extend([target["cache_dir"], target["offload_folder"]])
    assert all(is_within(path, content) for path in paths)
