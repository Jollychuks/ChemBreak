from chembreak16.policy import QPolicy

def settings():
    return {'discount':0.9,'global_learning_rate':0.30,'context_learning_rate':0.25,'task_learning_rate':0.15,'global_weight':0.45,'hc_weight':0.15,'hd_weight':0.15,'ot_weight':0.15,'task_weight':0.10,'novel_state_epsilon_bonus':0.10,'negative_feedback_epsilon_bonus':0.05,'max_effective_epsilon':0.35,'repeat_nonpositive_penalty':0.75,'hard_block_after_nonpositive_repeats':2}
def keys(g='G',t='T',hc='HC1',hd='HD1',ot='OT1'): return {'global':g,'task':t,'hc':hc,'hd':hd,'ot':ot}

def test_hierarchical_tables_update_at_distinct_levels_and_freeze(tmp_path):
    p=QPolicy(settings(),16); d=p.select('TASK1',keys(),1.0,['CONTINUE_CONTEXT','REFINE_SCOPE']); a=d['action']; p.update('TASK1',keys(),a,1.0,keys('G2','T2'),True)
    assert p.updates==1 and p.q_global['G'][a]>0 and p.q_hc['HC1']['G'][a]>0 and p.q_task['TASK1']['T'][a]>0
    f=tmp_path/'f.json'; p.freeze(f); q=QPolicy.load(f,settings(),16); assert q.frozen and q.to_dict()['policy_schema_version']==3

def test_global_knowledge_transfers_to_unseen_task_without_task_memory():
    p=QPolicy(settings(),16); a='REFINE_SCOPE'; p.update('TRAIN',keys(),a,1.0,keys('G2','T2'),True)
    score,vals,visits,active=p.combined('UNSEEN',keys(t='UNSEEN_T'),a)
    assert score>0 and 'global' in active and visits['task']==0 and vals['task']==0

def test_weighted_mean_does_not_sum_duplicate_evidence():
    p=QPolicy(settings(),16); a='REFINE_SCOPE'; p.update('TASK1',keys(),a,1.0,keys('G2','T2'),True)
    score,vals,visits,active=p.combined('TASK1',keys(),a)
    assert min(vals[k] for k in active)<=score<=max(vals[k] for k in active)
    assert score < sum(vals[k] for k in active)

def test_cold_start_is_labeled_and_repetition_block_works():
    p=QPolicy(settings(),16); d=p.select('X',keys(),0.0,['REFINE_SCOPE','DECOMPOSE_GOAL']); assert d['mode']=='cold_start'
    recent=[{'action':'REFINE_SCOPE','reward':-0.3,'success':False},{'action':'REFINE_SCOPE','reward':-0.2,'success':False}]
    d=p.select('X',keys(),0.0,['REFINE_SCOPE','DECOMPOSE_GOAL'],recent=recent); assert 'REFINE_SCOPE' in d['blocked_actions'] and d['action']=='DECOMPOSE_GOAL'

def test_adaptive_epsilon_is_bounded():
    p=QPolicy(settings(),16); d=p.select('X',keys(),0.30,['CONTINUE_CONTEXT','REFINE_SCOPE'],recent=[{'action':'CONTINUE_CONTEXT','reward':-0.1,'success':False}]); assert d['effective_epsilon']==0.35
