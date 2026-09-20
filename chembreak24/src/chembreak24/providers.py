from __future__ import annotations
import json, time
from .constants import ACTIONS, RESPONSE_CLASSES, CHCS_REASON_CODES

ATTACK_SCHEMA={
    'type':'object',
    'properties':{
        'action_id':{'type':'string','enum':list(ACTIONS)},
        'reason_code':{'type':'string'},
        'utterance':{'type':'string'},
    },
    'required':['action_id','reason_code','utterance'],
    'additionalProperties':False,
}
CANDIDATE_SCHEMA={
    'type':'object',
    'properties':{
        'goal_preservation':{'type':'number','minimum':0,'maximum':1},
        'task_fidelity':{'type':'number','minimum':0,'maximum':1},
        'drifted':{'type':'boolean'},
        'confidence':{'type':'number','minimum':0,'maximum':1},
        'reason_code':{'type':'string'},
    },
    'required':['goal_preservation','task_fidelity','drifted','confidence','reason_code'],
    'additionalProperties':False,
}
CHCS_RESPONSE_SCHEMA={
    'type':'object',
    'properties':{
        'chcs':{'type':'integer','minimum':1,'maximum':5},
        'response_class':{'type':'string','enum':list(RESPONSE_CLASSES)},
        'confidence':{'type':'number','minimum':0,'maximum':1},
        'reason_code':{'type':'string','enum':list(CHCS_REASON_CODES)},
    },
    'required':['chcs','response_class','confidence','reason_code'],
    'additionalProperties':False,
}

class StructuredOutputError(RuntimeError):pass
class ProviderPolicyBlock(RuntimeError):
    def __init__(self,provider,code,message):
        self.provider=str(provider); self.code=str(code or 'provider_policy'); self.provider_message=str(message or '')
        super().__init__(f'{self.provider} policy block ({self.code})')

def _clean(text):
    text=str(text or '').strip()
    if text.startswith('```'):
        lines=text.splitlines(); lines=lines[1:] if lines else lines
        lines=lines[:-1] if lines and lines[-1].strip()=='```' else lines
        text='\n'.join(lines).strip()
    return text

def _coerce(value):
    if isinstance(value,dict):return value
    if hasattr(value,'model_dump'):
        x=value.model_dump(); return x if isinstance(x,dict) else None
    if isinstance(value,str):
        try:
            x=json.loads(value); return x if isinstance(x,dict) else None
        except Exception:return None
    return None

def _validate(data,schema):
    if not isinstance(data,dict):raise ValueError('Structured output must be object')
    for k in schema['required']:
        if k not in data:raise ValueError(f'Missing required field {k}')
    extra=set(data)-set(schema['properties'])
    if schema.get('additionalProperties') is False and extra:raise ValueError(f'Unexpected fields {sorted(extra)}')
    for k,s in schema['properties'].items():
        v=data[k]; typ=s.get('type')
        if typ=='string' and (not isinstance(v,str) or not v.strip()):raise ValueError(f'{k} must be nonempty string')
        if typ=='boolean' and not isinstance(v,bool):raise ValueError(f'{k} must be boolean')
        if typ=='integer' and (not isinstance(v,int) or isinstance(v,bool)):raise ValueError(f'{k} must be integer')
        if typ=='number' and (not isinstance(v,(int,float)) or isinstance(v,bool)):raise ValueError(f'{k} must be numeric')
        if 'enum' in s and v not in s['enum']:raise ValueError(f'{k} invalid enum')
        if typ in {'number','integer'} and (v<s.get('minimum',v) or v>s.get('maximum',v)):raise ValueError(f'{k} outside range')
    return data

def _schema_clean(x):
    if isinstance(x,dict):return {k:_schema_clean(v) for k,v in x.items() if k!='propertyOrdering'}
    if isinstance(x,list):return [_schema_clean(v) for v in x]
    return x

def _decode_vertex(response):
    parsed=_coerce(getattr(response,'parsed',None))
    if parsed is not None:return parsed
    candidates=getattr(response,'candidates',None) or []
    reasons=[str(getattr(c,'finish_reason','')) for c in candidates]
    upper=' '.join(reasons).upper()
    if any(x in upper for x in ('SAFETY','BLOCKLIST','PROHIBITED_CONTENT')):
        raise ProviderPolicyBlock('google',reasons[0] if reasons else 'safety','Gemini response was safety-blocked')
    text=_clean(getattr(response,'text','') or '')
    if not text:raise StructuredOutputError(f'Gemini returned no structured output; finish_reason={reasons}')
    try:return json.loads(text)
    except Exception as exc:raise StructuredOutputError(f'Invalid Gemini JSON: {exc}') from exc

class MockRoles:
    def attack(self,prompt,action):return {'action_id':action,'reason_code':'mock','utterance':f'[MOCK {action} HIGH-LEVEL TEST MESSAGE]'}
    def judge_candidate(self,prompt,candidate):return {'goal_preservation':1.0,'task_fidelity':1.0,'drifted':False,'confidence':.99,'reason_code':'mock_match'}
    def judge_response(self,prompt,response):return {'chcs':1,'response_class':'refusal','confidence':.99,'reason_code':'REFUSAL'}

class VertexStructuredLLM:
    def __init__(self,project_id,cfg,client=None):
        self.project_id=project_id; self.cfg=cfg; self._client_override=client; self._clients={}
    def _client(self,loc):
        if self._client_override is not None:return self._client_override
        if loc not in self._clients:
            from google import genai
            self._clients[loc]=genai.Client(vertexai=True,project=self.project_id,location=loc)
        return self._clients[loc]
    def _config(self,types,system,schema,attempt):
        base=int(self.cfg.get('max_output_tokens',2048)); cap=int(self.cfg.get('retry_max_output_tokens',4096))
        kwargs={
            'system_instruction':system,
            'max_output_tokens':min(cap,base*(2**(attempt-1))),
            'response_mime_type':'application/json',
            'seed':int(self.cfg.get('seed',23023))+(attempt-1)*int(self.cfg.get('retry_seed_step',1)),
        }
        if self.cfg.get('temperature') is not None:kwargs['temperature']=float(self.cfg['temperature'])
        level=self.cfg.get('thinking_level')
        if level:kwargs['thinking_config']=types.ThinkingConfig(thinking_level=str(level))
        try:return types.GenerateContentConfig(**kwargs,response_json_schema=_schema_clean(schema))
        except Exception:return types.GenerateContentConfig(**kwargs,response_schema=_schema_clean(schema))
    def _call(self,system,prompt,schema):
        from google.genai import types
        attempts=max(1,int(self.cfg.get('attempts',5))); backoff=max(0,float(self.cfg.get('retry_backoff_seconds',1))); last=None
        client=self._client(self.cfg.get('location','global'))
        for attempt in range(1,attempts+1):
            try:
                resp=client.models.generate_content(model=self.cfg['model'],contents=prompt,config=self._config(types,system,schema,attempt))
                data=_decode_vertex(resp); return _validate(data,schema)
            except ProviderPolicyBlock:raise
            except Exception as exc:
                last=exc
                if attempt<attempts:time.sleep(min(backoff*(2**(attempt-1)),12))
        if isinstance(last,StructuredOutputError):raise last
        raise RuntimeError(f'Gemini structured call failed after {attempts} attempts: {type(last).__name__}: {last}') from last

class GeminiAttackLLM(VertexStructuredLLM):
    def attack(self,prompt,action):
        from .prompts import ATTACK_LLM_SYSTEM
        data=self._call(ATTACK_LLM_SYSTEM,prompt,ATTACK_SCHEMA)
        if data['action_id']!=action:raise ValueError(f'Attack LLM changed selected action {action} to {data["action_id"]}')
        return data

class GeminiIntentGateLLM(VertexStructuredLLM):
    def judge_candidate(self,prompt,candidate):
        from .prompts import CANDIDATE_JUDGE_SYSTEM
        return self._call(CANDIDATE_JUDGE_SYSTEM,prompt,CANDIDATE_SCHEMA)

class GeminiChcsJudge(VertexStructuredLLM):
    def judge_response(self,prompt,response):
        from .prompts import CHCS_JUDGE_SYSTEM
        return self._call(CHCS_JUDGE_SYSTEM,prompt,CHCS_RESPONSE_SCHEMA)

class Roles:
    def __init__(self,attack,intent_gate,chcs_judge):
        self.attack_llm=attack; self.intent_gate_llm=intent_gate; self.chcs_judge_llm=chcs_judge
    def attack(self,prompt,action):return self.attack_llm.attack(prompt,action)
    def judge_candidate(self,prompt,candidate):return self.intent_gate_llm.judge_candidate(prompt,candidate)
    def judge_response(self,prompt,response):return self.chcs_judge_llm.judge_response(prompt,response)

def make_roles(config,project_id):
    if config['run']['dry_run']:return MockRoles()
    return Roles(
        GeminiAttackLLM(project_id,config['roles']['attack_llm']),
        GeminiIntentGateLLM(project_id,config['roles']['intent_gate_llm']),
        GeminiChcsJudge(project_id,config['roles']['chcs_judge_llm']),
    )
