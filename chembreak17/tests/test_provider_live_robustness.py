import json, sys, types as pytypes
from types import SimpleNamespace
import pytest
from chembreak17.providers import JUDGE_SCHEMA, ATTACK_SCHEMA, StructuredOutputError, VertexJudgeLLM, OpenAIAttackLLM, _decode_vertex_structured_response, _validate

VALID_JUDGE={'response_class':'refusal','goal_progress':0.0,'task_fidelity':1.0,'chemistry_relevance':0.2,'policy_violation':0,'actionable_harm':0,'success':False,'confidence':0.99}
class FakeResponse:
    def __init__(self,*,parsed=None,text=None,finish_reason='STOP',thoughts=0):
        self.parsed=parsed; self.text=text; self.candidates=[SimpleNamespace(finish_reason=finish_reason)]; self.usage_metadata=SimpleNamespace(prompt_token_count=100,candidates_token_count=40,thoughts_token_count=thoughts,total_token_count=140+thoughts)

def test_vertex_decoder_prefers_parsed_and_rejects_truncation():
    assert _decode_vertex_structured_response(FakeResponse(parsed=VALID_JUDGE,text='{}'))==VALID_JUDGE
    with pytest.raises(StructuredOutputError,match='Invalid or truncated JSON'):_decode_vertex_structured_response(FakeResponse(text='{"response_class":"refusal","goal_progress":"',finish_reason='MAX_TOKENS',thoughts=900))

def test_vertex_generation_config_escalates_tokens():
    class TC:
        def __init__(self,**kwargs):self.kwargs=kwargs
    class GC:
        def __init__(self,**kwargs):self.kwargs=kwargs
    t=SimpleNamespace(ThinkingConfig=TC,GenerateContentConfig=GC); cfg={'temperature':0.0,'max_output_tokens':900,'retry_max_output_tokens':4096,'seed':13,'retry_seed_step':1,'thinking_budget':0}
    c1=VertexJudgeLLM._make_generation_config(t,cfg,'system',JUDGE_SCHEMA,1); c3=VertexJudgeLLM._make_generation_config(t,cfg,'system',JUDGE_SCHEMA,3); assert c1.kwargs['max_output_tokens']==900 and c3.kwargs['max_output_tokens']==3600 and c1.kwargs['thinking_config'].kwargs=={'thinking_budget':0}

def test_openai_attack_llm_uses_structured_output_and_retries():
    good={'action_id':'REPHRASE_GOAL','reason':'mock','utterance':'safe high-level message'}
    class Responses:
        def __init__(self):self.calls=[]
        def create(self,**kwargs):
            self.calls.append(kwargs)
            if len(self.calls)==1:raise RuntimeError('temporary')
            return SimpleNamespace(output_text=json.dumps(good),status='completed',incomplete_details=None)
    client=SimpleNamespace(responses=Responses()); llm=OpenAIAttackLLM({'model':'gpt-5.6-sol','reasoning_effort':'low','max_output_tokens':600,'retry_max_output_tokens':1200,'attempts':2,'retry_backoff_seconds':0},client=client)
    out=llm.attack('prompt','REPHRASE_GOAL'); assert out==good and len(client.responses.calls)==2
    call=client.responses.calls[-1]; assert call['model']=='gpt-5.6-sol' and call['text']['format']['type']=='json_schema' and call['text']['format']['strict'] is True and call['store'] is False
    _validate(out,ATTACK_SCHEMA)
