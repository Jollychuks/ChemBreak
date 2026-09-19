from chembreak21.providers import MockRoles,ATTACK_SCHEMA,CANDIDATE_SCHEMA,RESPONSE_SCHEMA,_validate

def test_mock_roles_shapes():
    r=MockRoles(); _validate(r.attack('x','REFINE_SCOPE'),ATTACK_SCHEMA); _validate(r.judge_candidate('x','y'),CANDIDATE_SCHEMA); _validate(r.judge_response('x','y'),RESPONSE_SCHEMA)
