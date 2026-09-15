import json
from chembreak15.policy import QPolicy


def settings():
    return {
        'learning_rate':0.5,'discount':0.9,'task_weight':0.75,
        'novel_state_epsilon_bonus':0.10,'negative_feedback_epsilon_bonus':0.05,'max_effective_epsilon':0.35,
        'repeat_nonpositive_penalty':0.75,'hard_block_after_nonpositive_repeats':2,
    }


def test_policy_updates_state_specific_task_memory_and_freezes(tmp_path):
    p=QPolicy(settings(),15)
    d=p.select('T1','S1',1.0,['CONTINUE_CONTEXT','REFINE_SCOPE'])
    p.update('T1','S1',d['action'],1.0,'S2',True)
    assert p.updates==1
    # Task memory learned in S1 must not bleed into a different state S_OTHER.
    assert p._qt('T1','S1',d['action']) > 0
    assert p._qt('T1','S_OTHER',d['action']) == 0
    f=tmp_path/'f.json'; p.freeze(f); q=QPolicy.load(f,settings(),15)
    assert q.frozen and q.to_dict()['policy_schema_version']==2


def test_repetition_penalty_and_hard_block_after_two_nonpositive_repeats():
    p=QPolicy(settings(),15)
    recent=[
        {'action':'REFINE_SCOPE','reward':-0.3,'success':False},
        {'action':'REFINE_SCOPE','reward':-0.2,'success':False},
    ]
    d=p.select('T1','S1',0.0,['REFINE_SCOPE','DECOMPOSE_GOAL'],recent=recent)
    assert 'REFINE_SCOPE' in d['blocked_actions']
    assert d['action']=='DECOMPOSE_GOAL'
    assert d['repeat_penalty']==0.0


def test_adaptive_epsilon_is_bounded_and_responds_to_novelty_and_feedback():
    p=QPolicy(settings(),15)
    # New state + immediate non-positive feedback would add .15, but cap at .35.
    d=p.select('T1','NEW',0.30,['CONTINUE_CONTEXT','REFINE_SCOPE'],recent=[{'action':'CONTINUE_CONTEXT','reward':-0.1,'success':False}])
    assert d['effective_epsilon']==0.35
    assert d['novel_state_bonus']==0.10
    assert d['negative_feedback_bonus']==0.05


def test_positive_repeat_is_not_blocked():
    p=QPolicy(settings(),15)
    recent=[
        {'action':'REFINE_SCOPE','reward':0.4,'success':False},
        {'action':'REFINE_SCOPE','reward':0.2,'success':False},
    ]
    d=p.select('T1','S1',0.0,['REFINE_SCOPE','DECOMPOSE_GOAL'],recent=recent)
    assert d['blocked_actions']==[]
