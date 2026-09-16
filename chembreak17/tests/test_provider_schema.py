import pytest
from chembreak17.providers import _validate, JUDGE_SCHEMA, ATTACK_SCHEMA

def test_judge_schema_rejects_out_of_range_values():
    data={'response_class':'refusal','goal_progress':0.0,'task_fidelity':1.0,'chemistry_relevance':0.2,'policy_violation':5,'actionable_harm':0,'success':False,'confidence':0.9}
    with pytest.raises(ValueError,match='policy_violation outside range'):_validate(data,JUDGE_SCHEMA)

def test_attack_schema_rejects_changed_action_label():
    data={'action_id':'NOT_REAL','reason':'x','utterance':'x'}
    with pytest.raises(ValueError,match='invalid value'):_validate(data,ATTACK_SCHEMA)
