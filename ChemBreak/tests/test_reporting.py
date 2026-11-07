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


if __name__ == "__main__":
    unittest.main()
