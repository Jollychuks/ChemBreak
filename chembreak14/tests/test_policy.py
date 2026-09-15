from chembreak14.policy import QPolicy
def test_policy_updates_and_freezes(tmp_path):
    s={'learning_rate':0.5,'discount':0.9,'task_weight':1.0}; p=QPolicy(s,14)
    a,mode,*_=p.select('T1','S1',1.0,['CONTINUE_CONTEXT','REFINE_SCOPE'])
    p.update('T1','S1',a,1.0,'S2',True)
    assert p.updates==1
    f=tmp_path/'f.json'; p.freeze(f); q=QPolicy.load(f,s,14); assert q.frozen
