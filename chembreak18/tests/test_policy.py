import json
from chembreak18.policy import QPolicy

def settings():return {'discount':0.9,'global_learning_rate':0.30,'context_learning_rate':0.25,'task_learning_rate':0.15,'global_weight':0.45,'hc_weight':0.15,'hd_weight':0.15,'ot_weight':0.15,'task_weight':0.10,'novel_state_epsilon_bonus':0.10,'negative_feedback_epsilon_bonus':0.05,'max_effective_epsilon':0.35,'repeat_nonpositive_penalty':0.75,'hard_block_after_nonpositive_repeats':2}
def keys(g='G',t='T',hc='HC1',hd='HD1',ot='OT1'):return {'global':g,'task':t,'hc':hc,'hd':hd,'ot':ot}

def test_hierarchical_tables_update_and_freeze(tmp_path):
    p=QPolicy(settings(),17); d=p.select('TASK1',keys(),1.0,['CONTINUE_CONTEXT','REFINE_SCOPE']); a=d['action']; p.update('TASK1',keys(),a,1.0,keys('G2','T2'),True)
    assert p.updates==1 and p.q_global['G'][a]>0 and p.q_task['TASK1']['T'][a]>0
    f=tmp_path/'f.json'; p.freeze(f); q=QPolicy.load(f,settings(),17); assert q.frozen and q.to_dict()['policy_schema_version']==5

def test_global_knowledge_transfers_across_tasks():
    p=QPolicy(settings(),17); a='REFINE_SCOPE'; p.update('TASK1',keys(),a,1.0,keys('G2','T2'),True); score,vals,visits,active=p.combined('TASK2',keys(t='OTHER'),a); assert score>0 and 'global' in active and visits['task']==0 and vals['task']==0

def test_cold_start_and_repetition_block():
    p=QPolicy(settings(),17); d=p.select('X',keys(),0.0,['REFINE_SCOPE','DECOMPOSE_GOAL']); assert d['mode']=='cold_start'
    recent=[{'action':'REFINE_SCOPE','reward':-0.3,'success':False},{'action':'REFINE_SCOPE','reward':-0.2,'success':False}]; d=p.select('X',keys(),0.0,['REFINE_SCOPE','DECOMPOSE_GOAL'],recent=recent); assert 'REFINE_SCOPE' in d['blocked_actions'] and d['action']=='DECOMPOSE_GOAL'

def test_frozen_selection_does_not_mutate_policy():
    p=QPolicy(settings(),17); p.update('T',keys(),'REFINE_SCOPE',1.0,keys('G2','T2'),True); p.frozen=True; before=p.to_dict(); before.pop('saved_at_utc',None); p.select('T',keys(),0.0,['REFINE_SCOPE','DECOMPOSE_GOAL']); after=p.to_dict(); after.pop('saved_at_utc',None); assert before==after


def test_exploitation_never_selects_unsupported_zero_action():
    p=QPolicy(settings(),17)
    supported='REFINE_SCOPE'; unseen='DECOMPOSE_GOAL'
    # Give the supported action negative learned value.  Exploitation must still
    # choose among supported actions rather than treating the unseen zero action
    # as if it were learned evidence.
    p.update('TASK1',keys(),supported,-1.0,keys('G2','T2'),True)
    d=p.select('TASK2',keys(t='OTHER'),0.0,[supported,unseen])
    assert d['mode']=='exploitation'
    assert d['action']==supported
    assert d['active_components']
