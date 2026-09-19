from chembreak20.policy import QPolicy
from chembreak20.constants import ACTIONS

def cfg():return {'discount':.9,'global_learning_rate':.3,'context_learning_rate':.25,'task_learning_rate':.15,'global_weight':.45,'hc_weight':.15,'hd_weight':.15,'ot_weight':.15,'task_weight':.1,'support_confidence_target_visits':3,'novel_state_epsilon_bonus':0,'negative_feedback_epsilon_bonus':0,'max_effective_epsilon':.35,'repeat_nonpositive_penalty':.75,'hard_block_after_nonpositive_repeats':2}
def keys():return {'global':'g','task':'t','hc':'h','hd':'d','ot':'o'}
def test_cold_start_not_exploitation():
    p=QPolicy(cfg(),1); d=p.select('T',keys(),0,ACTIONS,[],{},[]); assert d['mode']=='cold_start'
def test_route_bonus_can_support_action():
    p=QPolicy(cfg(),1); d=p.select('T',keys(),0,ACTIONS,[],{'REPHRASE_GOAL':.5},[]); assert d['mode']=='exploitation' and d['action']=='REPHRASE_GOAL'
def test_update_creates_support():
    p=QPolicy(cfg(),1); p.update('T',keys(),'REFINE_SCOPE',1,keys(),True); q,vals,vis,active=p.combined('T',keys(),'REFINE_SCOPE'); assert active and sum(vis.values())>0
def test_frozen_update_forbidden():
    p=QPolicy(cfg(),1); p.frozen=True
    import pytest
    with pytest.raises(RuntimeError):p.update('T',keys(),'REFINE_SCOPE',1,keys(),True)
