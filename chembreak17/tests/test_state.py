from chembreak17.state import EpisodeState

def test_global_state_is_taxonomy_free_and_reusable():
    t={'assignment_id':'A','hc_id':'HC1','hd_id':'HD1','ot_id':'OT1'}; b={'response_class':'refusal','goal_progress':0.2,'task_fidelity':0.9}
    s=EpisodeState.initial(t,b,4); keys=s.policy_keys()
    assert 'HC1' not in keys['global'] and 'HD1' not in keys['global'] and 'OT1' not in keys['global']
    assert keys['hc']=='HC1' and keys['hd']=='HD1' and keys['ot']=='OT1'
    assert 'stage=start' in keys['global'] and 'resp=refusal' in keys['task']
    s.advance('REFINE_SCOPE','p','r',{'response_class':'partial_compliance','goal_progress':0.4,'task_fidelity':0.9,'success':False},0.2)
    k2=s.policy_keys(); assert 'trend=improved' in k2['global'] and 'reward=positive' in k2['global']
