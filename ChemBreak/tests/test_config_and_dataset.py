import copy
import tempfile
import unittest
from pathlib import Path

import yaml

from chembreak.config import load_config, validate_config
from chembreak.cli import _runtime_config
from chembreak.runner import ChemBreakRunner
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
        self.assertEqual(result["dataset_profile"], "prompts28")

    def test_locked_chemharm500_dataset(self):
        result = verify_bundle(
            ROOT / "data" / "chemharm500_tasks.xlsx",
            ROOT / "data" / "chemharm500_manifest.csv",
            ROOT / "data" / "chemharm500_lock.json",
        )
        self.assertEqual(result["tasks"], 500)
        self.assertEqual(result["dataset_profile"], "chemharm500")

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
        self.assertFalse(config["experiment"]["hybrid_context_across_stages"])
        self.assertEqual(config["experiment"]["stage_1_strategy"], "adaptive_actions")
        self.assertEqual(config["experiment"]["stage_2_strategy"], "progressive_buildup_attempt_a")
        self.assertEqual(
            config["experiment"]["stage_3_strategy"],
            "progressive_buildup_attempt_b_if_stage_2_fails",
        )
        self.assertFalse(config["experiment"]["progressive_stage_context_restart"])
        self.assertEqual(config["experiment"]["complete_objective_turn"], 5)
        self.assertEqual(config["experiment"]["complete_objective_recovery_turn"], 6)

    def test_stage_one_gate_settings_match_v1_3(self):
        config = yaml.safe_load((ROOT / "configs" / "config.gpt.yaml").read_text())
        self.assertEqual(
            config["roles"]["intent_gate_llm"],
            {
                "provider": "vertex_genai",
                "model": "gemini-3.8-flash",
                "location": "global",
                "thinking_level": "low",
                "max_output_tokens": 1024,
                "retry_max_output_tokens": 2048,
                "attempts": 2,
                "retry_backoff_seconds": 1.0,
                "retry_seed_step": 1,
                "timeout_seconds": 45,
                "seed": 34034,
            },
        )
        self.assertEqual(config["roles"]["progressive_gate_llm"]["model"], "gemini-3.1-pro-preview")

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

    def test_runtime_dataset_switch_selects_chemharm500(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "runtime" / "chemharm500.yaml"
            _runtime_config(
                ROOT / "configs" / "config.gpt.yaml", runtime, None, None, "chemharm500"
            )
            config = load_config(runtime)
            self.assertEqual(config["run"]["dataset_profile"], "chemharm500")
            self.assertEqual(config["experiment"]["task_count"], 500)
            self.assertEqual(
                Path(config["run"]["prompts_path"]),
                (ROOT / "data" / "chemharm500_tasks.xlsx").resolve(),
            )
            self.assertTrue(validate_config(config))

    def test_chemharm500_uses_the_same_dry_run_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            config = yaml.safe_load((ROOT / "configs" / "config.gpt.yaml").read_text())
            config["run"].update({
                "dataset_profile": "chemharm500",
                "experiment_revision": "CHEMBREAK_PROGRESSIVE_RHS_OFS_CHEMHARM500_V1",
                "run_id": "chemharm500_architecture_test",
                "task_limit": 1,
                "live_progress": False,
                "prompts_path": str((ROOT / "data" / "chemharm500_tasks.xlsx").resolve()),
                "manifest_path": str((ROOT / "data" / "chemharm500_manifest.csv").resolve()),
                "lock_path": str((ROOT / "data" / "chemharm500_lock.json").resolve()),
                "output_root": str(temporary / "runs"),
                "artifact_root": str(temporary / "artifacts"),
            })
            config["experiment"]["task_count"] = 500
            config_path = temporary / "config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            runner = ChemBreakRunner(config_path, "ChemDFM")
            try:
                runner.run_all()
                queries = runner.store.target_queries()
                turns = runner.store.turns()
            finally:
                runner.close()
            self.assertEqual(len(queries), 19)
            self.assertEqual(len([row for row in turns if row["phase"] == "adaptive"]), 18)


if __name__ == "__main__":
    unittest.main()
