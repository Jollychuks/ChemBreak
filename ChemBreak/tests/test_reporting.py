import contextlib
import io
import unittest

from chembreak.reporting import Reporter


class ReportingTests(unittest.TestCase):
    def test_detailed_text_is_limited(self):
        reporter = Reporter(
            "ChemDFM",
            28,
            enabled=True,
            settings={"detailed": True, "text_limit": 1500},
            judge_model="gpt-4o",
        )
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            reporter._block("TARGET RESPONSE", "x" * 1800)
        output = stream.getvalue()
        self.assertIn("[TRUNCATED after 1500 characters]", output)
        self.assertNotIn("x" * 1501, output)

    def test_compact_output_has_no_percentage_label(self):
        reporter = Reporter(
            "ChemDFM",
            28,
            enabled=True,
            settings={"detailed": False, "text_limit": 1500},
            judge_model="gpt-4o",
        )
        result = {
            "terminal_ofs5": False,
            "rhs": {"score": 2, "decision_status": "valid"},
            "ofs": {"score": 1, "decision_status": "valid"},
        }
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            reporter.result(1, 1, 1, result)
        output = stream.getvalue()
        self.assertIn("OFS_5_TASKS: 0/28", output)
        self.assertNotIn("%", output)

    def test_progressive_display_shows_strategy_level_route_and_both_judges(self):
        reporter = Reporter(
            "ChemDFM",
            28,
            enabled=True,
            settings={"detailed": True, "text_limit": 1500},
            judge_model="gpt-4o",
        )
        judgment = {
            "terminal_ofs5": False,
            "response_count": 1,
            "calibration_flags": [],
            "rhs": {
                "score": 2, "decision_status": "valid", "reason_code": "TEST",
                "scientific_credibility": "uncertain",
            },
            "ofs": {
                "score": 1, "decision_status": "valid", "reason_code": "TEST",
                "response_class": "hard_refusal",
            },
        }
        gate = {
            "goal_preservation": 0.3,
            "task_fidelity": 0.4,
            "action_alignment": 0.9,
            "trajectory_fidelity": 0.95,
            "level_compliance": 0.96,
            "intensity_ceiling_compliance": 0.97,
            "route_difference": 0.92,
        }
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            reporter.stage_start(
                1, "TASK", 2,
                {
                    "mode": "RESTART_CONVERSATION", "reason": "progressive_attempt_a_starts_fresh",
                    "strategy_mode": "progressive_buildup", "progressive_attempt": 1,
                },
                "TASK:adaptive_context:2",
            )
            reporter.turn_start(
                1, "TASK", 2, 1, 1, "TASK:adaptive_context:2",
                strategy_mode="progressive_buildup", progressive_attempt=1,
                buildup_level=0, trajectory_role="benign_domain_entry",
            )
            reporter.candidate_ready(
                "BENIGN_ENTRY", "candidate", gate, {"planner": 0.1, "gate": 0.2},
                strategy_mode="progressive_buildup", progressive_attempt=1,
                buildup_level=0, buildup_route="route alpha",
            )
            reporter.target_start("adaptive", 1, "TASK", 2, 1, "BENIGN_ENTRY", "TASK:adaptive_context:2")
            reporter.target_end("adaptive", "TASK", 2, 1, "target response", 0.3)
            reporter.judge_start("adaptive", "rhs")
            reporter.judge_end("adaptive", "rhs", judgment["rhs"], 0.1)
            reporter.judge_start("adaptive", "ofs")
            reporter.judge_end("adaptive", "ofs", judgment["ofs"], 0.1)
            reporter.result(
                1, 2, 1, judgment,
                {
                    "assignment_id": "TASK", "action": "BENIGN_ENTRY",
                    "strategy_mode": "progressive_buildup", "progressive_attempt": 1,
                    "buildup_level": 0, "buildup_route": "route alpha",
                    "response_class": "hard_refusal", "reward": -0.55,
                    "peak_rhs": 2, "peak_ofs": 1, "stagnation": 1,
                    "context_id": "TASK:adaptive_context:2", "timings": {},
                },
            )
        output = stream.getvalue()
        self.assertIn("strategy=progressive_buildup", output)
        self.assertIn("buildup_level=0/5", output)
        self.assertIn("route=route alpha", output)
        self.assertIn("ATTACK CANDIDATE", output)
        self.assertIn("TARGET RESPONSE", output)
        self.assertIn("RHS JUDGE END", output)
        self.assertIn("OFS JUDGE END", output)


if __name__ == "__main__":
    unittest.main()
