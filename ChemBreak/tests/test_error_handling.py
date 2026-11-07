import unittest

from chembreak.providers import StructuredOutputError, classify_judge_error


class FakeBadRequest(Exception):
    status_code = 400
    code = "bio_policy"
    body = {"error": {"code": "bio_policy", "message": "flagged for possible biological risk"}}


class ErrorHandlingTests(unittest.TestCase):
    def test_biological_policy_is_distinct(self):
        result = classify_judge_error(FakeBadRequest("possible biological risk"), "openai", "rhs", 1)
        self.assertEqual(result["category"], "biological_policy_refusal")
        self.assertEqual(result["status"], 400)
        self.assertIn("biological", result["message"])

    def test_malformed_json_category(self):
        error = StructuredOutputError("malformed_json", "Invalid JSON", {"finish_reason": "stop"})
        result = classify_judge_error(error, "local_qwen", "ofs", 1)
        self.assertEqual(result["category"], "malformed_json")
        self.assertEqual(result["metadata"]["structured_metadata"]["finish_reason"], "stop")

    def test_timeout_category(self):
        result = classify_judge_error(TimeoutError("request timed out"), "openai", "ofs", 2)
        self.assertEqual(result["category"], "timeout")


if __name__ == "__main__":
    unittest.main()
