import json
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import yaml

from chembreak6 import runner
from chembreak6.config import canonical_config, load_config
from chembreak6.providers import StructuredOutputError


ROOT = Path(__file__).resolve().parents[1]


def _mock_config(root: Path) -> Path:
    config = canonical_config(load_config(ROOT / "configs" / "config.test.yaml"))
    config["run"]["project_root"] = str(ROOT)
    config["run"]["output_root"] = str(root / "runs")
    config["run"]["task_bank_path"] = str(ROOT / "data" / "final_task_bank.csv")
    storage = config["storage"]
    storage.update(
        {
            "content_root": str(root),
            "require_content_routing": False,
            "require_separate_mount": False,
            "minimum_free_gb": 0,
            "storage_root": str(root),
            "hf_home": str(root / "cache" / "hf"),
            "hf_hub_cache": str(root / "cache" / "hf" / "hub"),
            "hf_modules_cache": str(root / "cache" / "hf" / "modules"),
            "xdg_cache_home": str(root / "cache" / "xdg"),
            "torch_home": str(root / "cache" / "torch"),
            "torchinductor_cache": str(root / "cache" / "torchinductor"),
            "triton_cache": str(root / "cache" / "triton"),
            "cuda_cache": str(root / "cache" / "cuda"),
            "pip_cache": str(root / "cache" / "pip"),
            "python_packages": str(root / "python_packages"),
            "temp_dir": str(root / "tmp"),
            "offload_dir": str(root / "offload"),
            "preflight_dir": str(root / "preflight"),
        }
    )
    for target in config["targets"]:
        target["cache_dir"] = storage["hf_hub_cache"]
        target["offload_folder"] = str(root / "offload" / target["id"])
    config_path = root / "runtime.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_separate_condition_runs_share_checkpoint_and_resume_without_duplicates():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config_path = _mock_config(root)
        run_dirs = [
            runner.run_condition(config_path, condition)
            for condition in (
                "C0_DIRECT",
                "C1_REPEATED_SINGLE",
                "C2_FIXED_MULTI",
                "C3_ADAPTIVE_MDP",
            )
        ]
        assert len(set(run_dirs)) == 1
        run_dir = run_dirs[0]
        status = runner.finalize_run(config_path, require_complete=True)
        assert status["complete"] is True
        with sqlite3.connect(run_dir / "state.sqlite3") as connection:
            counts_before = {
                "episodes": connection.execute("SELECT COUNT(*) FROM episodes").fetchone()[0],
                "transcripts": connection.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0],
                "evaluations": connection.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0],
                "assets": connection.execute("SELECT COUNT(*) FROM assets").fetchone()[0],
                "api_calls": connection.execute("SELECT COUNT(*) FROM api_calls").fetchone()[0],
            }
        assert counts_before == {
            "episodes": 96,
            "transcripts": 336,
            "evaluations": 336,
            "assets": 16,
            "api_calls": 832,
        }
        c0 = pd.read_csv(
            run_dir / "release" / "by_condition" / "C0_DIRECT_episode_results.csv"
        )
        assert len(c0) == 24
        assert set(c0["queries_used"]) == {1}
        adaptive = pd.read_csv(run_dir / "release" / "adaptive_mdp_metrics.csv")
        assert int(adaptive["episodes_with_feedback_opportunity"].sum()) == 24
        runner.run_condition(config_path, "C0_DIRECT")
        with sqlite3.connect(run_dir / "state.sqlite3") as connection:
            counts_after = {
                "episodes": connection.execute("SELECT COUNT(*) FROM episodes").fetchone()[0],
                "transcripts": connection.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0],
                "evaluations": connection.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0],
                "assets": connection.execute("SELECT COUNT(*) FROM assets").fetchone()[0],
                "api_calls": connection.execute("SELECT COUNT(*) FROM api_calls").fetchone()[0],
            }
        assert counts_after == counts_before
        run_status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
        assert run_status["complete"] is True


def test_recovery_cell_judges_saved_responses_without_new_target_queries():
    class _FailingSafetyClients:
        def __init__(self, config, project_id):
            self.history = []

        def call_json(self, role, prompt, system, call_role=None, **kwargs):
            raise StructuredOutputError("safety_judge", 3, ValueError("simulated truncation"))

        def drain_call_history(self):
            history = self.history
            self.history = []
            return history

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config_path = _mock_config(root)
        original_clients = runner.RoleClients
        runner.RoleClients = _FailingSafetyClients
        try:
            run_dir = runner.run_condition(config_path, "C0_DIRECT")
        finally:
            runner.RoleClients = original_clients
        with sqlite3.connect(run_dir / "state.sqlite3") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM episodes WHERE status='pending_judgment'"
            ).fetchone()[0] == 24
            assert connection.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0] == 24
            assert connection.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0] == 0
        recovery = runner.recover_pending_judgments(config_path, "C0_DIRECT")
        assert recovery["recovered"] == 24
        assert recovery["still_pending"] == 0
        with sqlite3.connect(run_dir / "state.sqlite3") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM episodes WHERE status='complete'"
            ).fetchone()[0] == 24
            assert connection.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0] == 24
            assert connection.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0] == 24
