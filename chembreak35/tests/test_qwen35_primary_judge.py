import json
from pathlib import Path
from unittest.mock import Mock

from chembreak35.providers import LocalQwenGGUFChcsJudge, FallbackChcsJudge, CHCS_RESPONSE_SCHEMA


def test_qwen_primary_schema_without_fallback():
    score = {'chcs': 3, 'response_class': 'partial_assistance', 'confidence': 0.8, 'reason_code': 'PARTIAL_ASSISTANCE'}
    completion = Mock()
    completion.choices = [Mock(message=Mock(content=json.dumps(score)), finish_reason='stop')]
    client = Mock()
    client.chat.completions.create.return_value = completion
    primary_cfg = {
        'model': 'ggml-org/Qwen3.6-35B-A3B-GGUF:Q4_K_M',
        'model_path': '/content/chembreak35_storage/models/Qwen3.6-35B-A3B-Q4_K_M.gguf',
        'attempts': 2, 'max_output_tokens': 1024, 'retry_max_output_tokens': 2048,
        'retry_backoff_seconds': 0,
    }
    primary = LocalQwenGGUFChcsJudge(primary_cfg, client=client)
    fallback = Mock()
    cascade = FallbackChcsJudge(primary, fallback, primary_cfg, {'model':'gemini-3.8-flash'})
    judged = cascade.judge_response('harmless sample assessment', '')
    assert judged['chcs'] == 3
    assert judged['fallback_used'] is False
    assert judged['final_judge'] == primary_cfg['model']
    fallback.judge_response.assert_not_called()
    assert client.chat.completions.create.call_args.kwargs['extra_body']['response_format']['schema'] == CHCS_RESPONSE_SCHEMA
    assert 'chat_template_kwargs' not in client.chat.completions.create.call_args.kwargs['extra_body']


def test_fallback_only_after_primary_invalid_schema():
    client = Mock()
    bad_completion = Mock()
    bad_completion.choices = [Mock(message=Mock(content='{"chcs":7}'), finish_reason='stop')]
    client.chat.completions.create.return_value = bad_completion
    primary_cfg = {'model':'ggml-org/Qwen3.6-35B-A3B-GGUF:Q4_K_M', 'attempts':2,'retry_backoff_seconds':0}
    primary = LocalQwenGGUFChcsJudge(primary_cfg, client=client)
    fallback = Mock()
    fallback.judge_response.return_value = {'chcs':2,'response_class':'limited_engagement','confidence':0.8,'reason_code':'MINIMAL_ASSISTANCE'}
    cascade = FallbackChcsJudge(primary, fallback, primary_cfg, {'model':'gemini-3.8-flash'})
    result = cascade.judge_response('harmless sample assessment','')
    assert result['fallback_used'] is True
    assert result['final_judge'] == 'gemini-3.8-flash'
    assert client.chat.completions.create.call_count == 2
    fallback.judge_response.assert_called_once()
