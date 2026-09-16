from chembreak19.state import EpisodeState

def test_global_state_is_taxonomy_free_reusable_and_fresh():
    t={'assignment_id':'A','hc_id':'HC1','hd_id':'HD1','ot_id':'OT1'}
    s=EpisodeState.initial(t,4); keys=s.policy_keys()
    assert s.response_class=='initial' and s.progress==0.0 and s.fidelity==1.0 and s.history==[]
    assert 'HC1' not in keys['global'] and 'HD1' not in keys['global'] and 'OT1' not in keys['global']
    assert keys['hc']=='HC1' and keys['hd']=='HD1' and keys['ot']=='OT1'
    assert 'stage=start' in keys['global'] and 'resp=initial' in keys['task']
    s.advance('REFINE_SCOPE','p','r',{'response_class':'partial_compliance','goal_progress':0.4,'task_fidelity':0.9,'success':False},0.2)
    k2=s.policy_keys(); assert 'trend=improved' in k2['global'] and 'reward=positive' in k2['global']
