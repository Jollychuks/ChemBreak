import json
from chembreak16.checkpoint import Store

def test_store_roundtrip_and_policy_snapshot(tmp_path):
    s=Store(tmp_path/'x.sqlite'); s.start_episode('baseline',0,'A')
    s.save_baseline('A','r',{'response_class':'refusal','goal_progress':0,'task_fidelity':1,'chemistry_relevance':0,'success':False})
    snapshot={'namespace':'CB16','policy_schema_version':3,'updates':1}; decision={'effective_epsilon':0.25,'q_global':0.1,'q_task':0.0,'blocked_actions':[]}
    s.save_turn_with_policy(policy_snapshot=snapshot,phase='learning',epoch=1,assignment_id='A',turn_index=1,action_id='REFINE_SCOPE',selection_mode='exploration',state_key='s',q_global=0.1,q_task=0.0,combined_q=0.1,prompt='p',response='r',judge={'success':False},decision=decision,reward=-0.1,latency_seconds=0.01)
    rows=s.get_turns('learning',1,'A'); assert len(rows)==1 and json.loads(rows[0]['decision_json'])['q_global']==0.1
    assert s.get_meta('training_policy_snapshot')==snapshot; s.close()
