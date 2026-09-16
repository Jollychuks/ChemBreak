import json
from chembreak19.checkpoint import Store

def test_store_roundtrip_and_learning_snapshots(tmp_path):
    s=Store(tmp_path/'x.sqlite'); s.start_episode('learning',1,'A')
    ps={'namespace':'CB19','policy_schema_version':4,'updates':1}; es={'namespace':'CB19','evidence_schema_version':1,'records':1}
    decision={'effective_epsilon':0.25,'q_global':0.1,'q_task':0.0,'blocked_actions':[]}
    s.save_turn_with_learning(policy_snapshot=ps,evidence_snapshot=es,phase='learning',epoch=1,assignment_id='A',turn_index=1,action_id='REFINE_SCOPE',selection_mode='exploration',state_key='s',q_global=0.1,q_task=0.0,combined_q=0.1,candidate_id='abc',realization_id='rid',candidate_source='attack_llm_fresh',candidate_rank_score=None,prompt='p',response='r',judge={'success':False},decision=decision,reward=-0.1,latency_seconds=0.01)
    rows=s.get_turns('learning',1,'A'); assert len(rows)==1 and rows[0]['candidate_id']=='abc' and rows[0]['realization_id']=='rid' and json.loads(rows[0]['decision_json'])['q_global']==0.1
    assert s.get_meta('training_policy_snapshot')==ps; assert s.get_meta('training_evidence_snapshot')==es; s.close()
