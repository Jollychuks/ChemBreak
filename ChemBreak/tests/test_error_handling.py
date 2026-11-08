import unittest

from chembreak.providers import ProviderPolicyBlock, Roles, StructuredOutputError, _decode_vertex, classify_judge_error


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

    def test_vertex_prompt_block_is_not_mislabeled_as_empty_output(self):
        class Feedback:
            block_reason = "PROHIBITED_CONTENT"
            block_reason_message = "Input was blocked"

        class Response:
            parsed = None
            prompt_feedback = Feedback()
            candidates = []
            text = ""
            model_version = "test-model"
            response_id = "response-1"

        with self.assertRaises(ProviderPolicyBlock) as raised:
            _decode_vertex(Response())
        self.assertEqual(raised.exception.code, "PROHIBITED_CONTENT")
        self.assertEqual(raised.exception.metadata["model_version"], "test-model")

    def test_vertex_empty_output_keeps_finish_diagnostics(self):
        class Candidate:
            finish_reason = "MAX_TOKENS"
            finish_message = "Output limit reached"
            safety_ratings = []

        class Response:
            parsed = None
            prompt_feedback = None
            candidates = [Candidate()]
            text = ""
            model_version = "test-model"
            response_id = "response-2"
            usage_metadata = {"output_tokens": 10}

        with self.assertRaises(StructuredOutputError) as raised:
            _decode_vertex(Response())
        self.assertEqual(raised.exception.category, "incomplete_output")
        self.assertIn("MAX_TOKENS", raised.exception.metadata["finish_reasons"][0])
        self.assertEqual(raised.exception.metadata["finish_messages"], ["Output limit reached"])

    def test_stage_one_and_progressive_gates_are_separate(self):
        class Gate:
            def __init__(self, name):
                self.name = name

            def judge_candidate(self, prompt, candidate):
                return {"gate": self.name}

        roles = Roles(object(), Gate("v1.3"), Gate("progressive"), object())
        self.assertEqual(roles.judge_candidate("p", "c", progressive=False)["gate"], "v1.3")
        self.assertEqual(roles.judge_candidate("p", "c", progressive=True)["gate"], "progressive")


if __name__ == "__main__":
    unittest.main()
