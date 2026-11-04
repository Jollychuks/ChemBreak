"""Harmless offline tests. No model/API calls, no operational source content generated."""
import json
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chembreak.providers import MockRoles, OpenAIOfsJudge, ProviderPolicyBlock
from chembreak.runner import ChemBreakRunner


def runner_for(root, kind):
    config = yaml.safe_load((ROOT / "configs/config.yaml").read_text())
    config["run"].update({
        "run_id": f"validation_{kind}",
        "task_limit": 1,
        "output_root": str(root / "runs"),
        "artifact_root": str(root / "artifacts"),
        "live_progress": False,
        "live_progress_show_text": False,
    })
    path = ROOT / "configs/_offline_validation.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    try:
        return ChemBreakRunner(path, "ChemDFM")
    finally:
        path.unlink(missing_ok=True)


class OFSBlocked(MockRoles):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def judge_response(self, prompt, response):
        self.calls += 1
        if self.calls >= 2:
            raise ProviderPolicyBlock("openai", "bio_policy", "Simulated provider restriction")
        return super().judge_response(prompt, response)


class RHSBlocked(MockRoles):
    def judge_rhs(self, prompt, response):
        raise ProviderPolicyBlock("openai", "bio_policy", "Simulated provider restriction")


class FakeBadRequest(RuntimeError):
    code = "bio_policy"
    status_code = 400
    body = {"error": {"code": "bio_policy"}}


def run_tests():
    with tempfile.TemporaryDirectory() as path:
        runner = runner_for(Path(path), "normal")
        try:
            summary = runner.run_all()
            assert summary["query_accounting"]["committed"] == 19
            assert summary["independent_score_coverage"]["both_valid"] == 19
            assert summary["evaluation_paused_tasks"] == 0
            assert len([e for e in runner.store.episodes() if e["phase"] == "adaptive"]) == 3
        finally:
            runner.close()
    with tempfile.TemporaryDirectory() as path:
        runner = runner_for(Path(path), "ofs_blocked")
        runner.roles = OFSBlocked()
        try:
            summary = runner.run_all()
            assert summary["query_accounting"]["committed"] == 2
            assert summary["judge_status_breakdown"]["ofs"] == {"valid": 1, "policy_blocked": 1}
            assert summary["evaluation_paused_tasks"] == 1
            assert [row["reward"] for row in runner.store.turns() if row["phase"] == "adaptive"] == [None]
            assert runner.store.episode_row("adaptive", 2, "CB-0001") is None
            assert (runner.out / "release/judgment_review_queue.csv").exists()
        finally:
            runner.close()
    with tempfile.TemporaryDirectory() as path:
        runner = runner_for(Path(path), "rhs_blocked")
        runner.roles = RHSBlocked()
        try:
            summary = runner.run_all()
            assert summary["query_accounting"]["committed"] == 19
            assert summary["independent_score_coverage"]["rhs_valid"] == 0
            assert summary["independent_score_coverage"]["ofs_valid"] == 19
            assert summary["evaluation_paused_tasks"] == 0
        finally:
            runner.close()
    assert OpenAIOfsJudge._policy_error_code(FakeBadRequest("Simulated restriction")) == "bio_policy"
    assert OpenAIOfsJudge._policy_error_code(RuntimeError("Bad input format")) is None
    print("PASS: normal 6x3; OFS-policy pause; RHS-only block; provider-error classification")


if __name__ == "__main__":
    run_tests()
