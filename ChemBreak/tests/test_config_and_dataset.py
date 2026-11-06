import copy
import tempfile
import unittest
from pathlib import Path

import yaml

from chembreak.config import load_config, validate_config
from chembreak.cli import _runtime_config
from chembreak.selection import verify_bundle


ROOT = Path(__file__).resolve().parents[1]


class ConfigAndDatasetTests(unittest.TestCase):
    def test_locked_dataset(self):
        result = verify_bundle(
            ROOT / "data" / "prompts.csv",
            ROOT / "data" / "prompts28_manifest.csv",
            ROOT / "data" / "prompts28_lock.json",
        )
        self.assertEqual(result["tasks"], 28)

    def test_gpt4o_is_default_and_openai_alias_is_switchable(self):
        config = load_config(ROOT / "configs" / "config.gpt.yaml")
        self.assertEqual(config["roles"]["judge_llm"]["model"], "gpt-4o")
        validate_config(config)
        for model in ("gpt-5.6-luna", "gpt-6-luna"):
            changed = copy.deepcopy(config)
            changed["roles"]["judge_llm"]["model"] = model
            self.assertTrue(validate_config(changed))

    def test_qwen_model_is_locked(self):
        config = load_config(ROOT / "configs" / "config.qwen.yaml")
        validate_config(config)
        changed = copy.deepcopy(config)
        changed["roles"]["judge_llm"]["model"] = "different-qwen"
        with self.assertRaises(ValueError):
            validate_config(changed)

    def test_three_stages_six_turns(self):
        config = yaml.safe_load((ROOT / "configs" / "config.gpt.yaml").read_text())
        self.assertEqual(config["experiment"]["adaptive_stages"], 3)
        self.assertEqual(config["experiment"]["max_turns_per_stage"], 6)
        self.assertEqual(config["experiment"]["max_target_queries_per_task"], 19)
        self.assertTrue(config["experiment"]["baseline_always_runs"])
        self.assertFalse(config["experiment"]["baseline_informs_adaptive"])

    def test_model_override_keeps_dataset_paths_portable(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "runtime" / "gpt-6-luna.yaml"
            _runtime_config(
                ROOT / "configs" / "config.gpt.yaml", runtime, "gpt-6-luna", "medium"
            )
            config = load_config(runtime)
            self.assertEqual(config["roles"]["judge_llm"]["model"], "gpt-6-luna")
            self.assertEqual(config["roles"]["judge_llm"]["reasoning_effort"], "medium")
            self.assertEqual(Path(config["run"]["prompts_path"]), (ROOT / "data" / "prompts.csv").resolve())
            self.assertTrue(validate_config(config))


if __name__ == "__main__":
    unittest.main()
