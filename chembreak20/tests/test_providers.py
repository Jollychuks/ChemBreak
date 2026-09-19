from chembreak20.providers import ATTACK_SCHEMA,CANDIDATE_SCHEMA,RESPONSE_SCHEMA,MockRoles

def test_response_schema_has_no_llm_success_boolean():assert 'success' not in RESPONSE_SCHEMA['properties']
def test_correctness_required():assert 'response_correctness' in RESPONSE_SCHEMA['required']
def test_goal_gate_fields():assert {'goal_preservation','task_fidelity','drifted'}<=set(CANDIDATE_SCHEMA['required'])
def test_attack_schema_action_fixed():assert 'action_id' in ATTACK_SCHEMA['required']
def test_mock_roles_shapes():
    m=MockRoles(); assert m.attack('{}','REFINE_SCOPE')['action_id']=='REFINE_SCOPE'; assert m.judge_candidate('{}','x')['drifted'] is False; assert m.judge_response('{}','x')['response_class']=='refusal'

def test_openai_explicit_refusal_is_detectable():
    from types import SimpleNamespace
    from chembreak20.providers import _openai_refusal
    r=SimpleNamespace(output=[SimpleNamespace(content=[SimpleNamespace(type='refusal',refusal='blocked')])])
    assert _openai_refusal(r)=='blocked'
