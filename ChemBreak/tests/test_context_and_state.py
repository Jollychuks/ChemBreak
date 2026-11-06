import json
import tempfile
import unittest
from pathlib import Path

import yaml

from chembreak.prompts import response_metric_prompt
from chembreak.providers import MockRoles
from chembreak.runner import ChemBreakRunner
from chembreak.scoring import combine_metric_results, validate_metric
from chembreak.state import InteractionState


TASK = {
    "assignment_id": "CB34P-TEST",
    "original_prompt": "Complete the locked test objective.",
    "goal_intent_anchor": "Complete the locked test objective.",
    "requested_outcome": "source_behavior",
    "behavior_sha256": "test",
}


def result(metric, score, response_class=None):
    common = {
        "score": score,
        "confidence": 0.9,
        "reason_code": "TEST",
        "rationale": "test",
        "scientific_credibility": "credible" if score >= 3 else "uncertain",
        "evidence": [{"context_turn": 1, "quote": "first response", "relevance": "test"}],
    }
    if metric == "rhs":
        common["response_class"] = response_class or "partial_assistance"
    else:
        common.update(
            {
                "objective_elements": ["element"],
                "fulfilled_elements": ["element"] if score == 5 else [],
                "critical_gaps": [] if score == 5 else ["element"],
            }
        )
    return validate_metric(metric, common, ["first response"])


class ContextAndStateTests(unittest.TestCase):
    def test_metric_prompt_is_cumulative_and_ordered(self):
        prompt = json.loads(response_metric_prompt("rhs", TASK, ["one", "two"], "ctx-1"))
        self.assertEqual(prompt["response_count"], 2)
        self.assertEqual([row["context_turn"] for row in prompt["target_responses"]], [1, 2])
        self.assertEqual([row["response"] for row in prompt["target_responses"]], ["one", "two"])

    def test_restart_resets_scoring_evidence_but_not_stage_turn_budget(self):
        state = InteractionState.initial(TASK, 1, 6)
        judgment = combine_metric_results(
            result("rhs", 3), result("ofs", 3), context_id="ctx-1", response_count=1, judge_model="test"
        )
        state.advance("REFINE_SCOPE", "candidate", "first response", judgment, 1.0)
        self.assertEqual(state.turn_index, 1)
        self.assertEqual(len(state.history), 2)
        state.restart_context()
        self.assertEqual(state.turn_index, 1)
        self.assertEqual(state.context_turn_index, 0)
        self.assertEqual(state.history, [])
        self.assertEqual((state.rhs, state.ofs), (1, 1))

    def test_partial_judgment_updates_only_valid_metric(self):
        state = InteractionState.initial(TASK, 1, 6)
        rhs_result = result("rhs", 3)
        from chembreak.scoring import unresolved_metric
        judgment = combine_metric_results(
            rhs_result, unresolved_metric("ofs", {"category": "timeout"}),
            context_id="ctx-1", response_count=1, judge_model="test",
        )
        state.advance("REFINE_SCOPE", "candidate", "first response", judgment, 0.2)
        self.assertEqual(state.rhs, 3)
        self.assertEqual(state.ofs, 1)
        self.assertEqual(state.ofs_status, "unresolved")

    def test_direct_baseline_never_skips_or_seeds_adaptive_mdp(self):
        class BaselineSuccessRoles(MockRoles):
            def judge_metric(self, metric, prompt):
                payload = json.loads(prompt)
                baseline = str(payload.get("context_id", "")).endswith(":baseline")
                if metric == "rhs":
                    return {
                        "score": 1, "confidence": 0.99, "reason_code": "TEST",
                        "rationale": "test", "response_class": "hard_refusal",
                        "scientific_credibility": "not_credible",
                        "evidence": [{"context_turn": 1, "quote": "cannot provide", "relevance": "test"}],
                    }
                score = 5 if baseline else 1
                return {
                    "score": score, "confidence": 0.99, "reason_code": "TEST",
                    "rationale": "test",
                    "scientific_credibility": "credible" if baseline else "not_credible",
                    "objective_elements": ["objective"],
                    "fulfilled_elements": ["objective"] if baseline else [],
                    "critical_gaps": [] if baseline else ["objective"],
                    "evidence": [{"context_turn": 1, "quote": "cannot provide", "relevance": "test"}],
                }

        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            config = yaml.safe_load((root / "configs" / "config.gpt.yaml").read_text())
            config["run"].update(
                {
                    "run_id": "baseline_independence_test",
                    "task_limit": 1,
                    "live_progress": False,
                    "prompts_path": str((root / "data" / "prompts.csv").resolve()),
                    "manifest_path": str((root / "data" / "prompts28_manifest.csv").resolve()),
                    "lock_path": str((root / "data" / "prompts28_lock.json").resolve()),
                    "output_root": str(temporary / "runs"),
                    "artifact_root": str(temporary / "artifacts"),
                }
            )
            config_path = temporary / "config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            runner = ChemBreakRunner(config_path, "ChemDFM")
            runner.roles = BaselineSuccessRoles()
            try:
                summary = runner.run_all()
                phases = [row["phase"] for row in runner.store.target_queries()]
            finally:
                runner.close()
            self.assertEqual(phases.count("baseline"), 1)
            self.assertEqual(phases.count("adaptive"), 18)
            self.assertEqual(summary["baseline_ofs_5_tasks"], 1)
            self.assertEqual(summary["adaptive_ofs_5_tasks"], 0)
            self.assertEqual(summary["ofs_5_tasks"], 1)


if __name__ == "__main__":
    unittest.main()
