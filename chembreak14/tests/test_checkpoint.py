from chembreak14.checkpoint import Store

def test_store_roundtrip_and_atomic_policy_snapshot(tmp_path):
    s=Store(tmp_path/'x.sqlite')
    s.start_episode('baseline',0,'A')
    s.save_baseline('A','r',{'response_class':'refusal','goal_progress':0,'task_fidelity':1,'chemistry_relevance':0,'success':False})
    assert s.baseline('A')['response_class']=='refusal'
    snapshot={'namespace':'CB14','updates':1}
    s.save_turn_with_policy(policy_snapshot=snapshot,phase='learning',epoch=1,assignment_id='A',turn_index=1,action_id='REFINE_SCOPE',selection_mode='exploration',state_key='s',q_general=0,q_task=0,combined_q=0,prompt='p',response='r',judge={'success':False},reward=-0.1,latency_seconds=0.01)
    assert len(s.get_turns('learning',1,'A'))==1
    assert s.get_meta('training_policy_snapshot')==snapshot
    s.close()
