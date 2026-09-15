import json
from types import SimpleNamespace

import pytest

from chembreak14.providers import (
    JUDGE_SCHEMA,
    StructuredOutputError,
    VertexRoles,
    _decode_structured_response,
    _validate,
)


VALID_JUDGE = {
    "response_class": "refusal",
    "goal_progress": 0.0,
    "task_fidelity": 1.0,
    "chemistry_relevance": 0.2,
    "policy_violation": 0,
    "actionable_harm": 0,
    "success": False,
    "confidence": 0.99,
}


class FakeResponse:
    def __init__(self, *, parsed=None, text=None, finish_reason="STOP", thoughts=0):
        self.parsed = parsed
        self.text = text
        self.candidates = [SimpleNamespace(finish_reason=finish_reason)]
        self.usage_metadata = SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=40,
            thoughts_token_count=thoughts,
            total_token_count=140 + thoughts,
        )


def test_structured_response_prefers_sdk_parsed_object():
    r = FakeResponse(parsed=VALID_JUDGE, text='{"this":"would not be used"}')
    assert _decode_structured_response(r) == VALID_JUDGE


def test_structured_response_accepts_fenced_json_text_fallback():
    text = "```json\n" + json.dumps(VALID_JUDGE) + "\n```"
    assert _decode_structured_response(FakeResponse(text=text)) == VALID_JUDGE


def test_truncated_json_is_never_silently_repaired():
    r = FakeResponse(text='{"response_class":"refusal","goal_progress":"', finish_reason="MAX_TOKENS", thoughts=900)
    with pytest.raises(StructuredOutputError, match="Invalid or truncated JSON") as exc:
        _decode_structured_response(r)
    message = str(exc.value)
    assert "MAX_TOKENS" in message
    assert "thoughts_token_count=900" in message


def test_decoded_object_still_passes_schema_validation():
    data = _decode_structured_response(FakeResponse(parsed=VALID_JUDGE))
    _validate(data, JUDGE_SCHEMA)


def test_generation_config_escalates_tokens_and_uses_json_schema():
    class FakeThinkingConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_types = SimpleNamespace(
        ThinkingConfig=FakeThinkingConfig,
        GenerateContentConfig=FakeGenerateContentConfig,
    )
    cfg = {
        "temperature": 0.0,
        "max_output_tokens": 900,
        "retry_max_output_tokens": 4096,
        "seed": 13,
        "retry_seed_step": 1,
        "thinking_budget": 0,
    }
    c1 = VertexRoles._make_generation_config(fake_types, cfg, "system", JUDGE_SCHEMA, 1)
    c3 = VertexRoles._make_generation_config(fake_types, cfg, "system", JUDGE_SCHEMA, 3)
    assert c1.kwargs["max_output_tokens"] == 900
    assert c3.kwargs["max_output_tokens"] == 3600
    assert "propertyOrdering" not in c1.kwargs["response_json_schema"]
    assert c1.kwargs["response_json_schema"]["required"] == JUDGE_SCHEMA["required"]
    assert c1.kwargs["thinking_config"].kwargs == {"thinking_budget": 0}

def test_vertex_call_retries_the_observed_truncated_json_failure(monkeypatch):
    import sys
    import types as pytypes

    class FakeThinkingConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_types = SimpleNamespace(
        ThinkingConfig=FakeThinkingConfig,
        GenerateContentConfig=FakeGenerateContentConfig,
    )
    fake_genai_module = pytypes.ModuleType('google.genai')
    fake_genai_module.types = fake_types
    fake_google_module = pytypes.ModuleType('google')
    fake_google_module.genai = fake_genai_module
    monkeypatch.setitem(sys.modules, 'google', fake_google_module)
    monkeypatch.setitem(sys.modules, 'google.genai', fake_genai_module)

    responses = [
        FakeResponse(text='{"response_class":"refusal","goal_progress":"', finish_reason='MAX_TOKENS', thoughts=900),
        FakeResponse(text='{"response_class":"refusal","goal_progress":"', finish_reason='MAX_TOKENS', thoughts=900),
        FakeResponse(text='{"response_class":"refusal","goal_progress":"', finish_reason='MAX_TOKENS', thoughts=900),
        FakeResponse(parsed=VALID_JUDGE),
    ]

    class FakeModels:
        def __init__(self):
            self.calls = 0
            self.max_tokens = []
        def generate_content(self, **kwargs):
            self.max_tokens.append(kwargs['config'].kwargs['max_output_tokens'])
            response = responses[self.calls]
            self.calls += 1
            return response

    models = FakeModels()
    fake_client = SimpleNamespace(models=models)
    roles = object.__new__(VertexRoles)
    roles._client = lambda _location: fake_client
    cfg = {
        'model':'gemini-2.5-flash','location':'global','temperature':0.0,
        'max_output_tokens':900,'retry_max_output_tokens':4096,
        'attempts':4,'retry_backoff_seconds':0,'retry_seed_step':1,'thinking_budget':0,
    }
    result = roles._call(cfg, 'system', 'prompt', JUDGE_SCHEMA)
    assert result == VALID_JUDGE
    assert models.calls == 4
    assert models.max_tokens == [900, 1800, 3600, 4096]
