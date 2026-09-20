from chembreak22.policy import QPolicy
from chembreak22.state import EpisodeState
from chembreak22.constants import ACTIONS

def cfg():return {'discount':.9,'global_learning_rate':.3,'task_learning_rate':.18,'global_weight':.55,'task_weight':.45,'support_confidence_target_visits':3,'novel_state_epsilon_bonus':.1,'negative_feedback_epsilon_bonus':.05,'max_effective_epsilon':.35,'repeat_nonpositive_penalty':.75,'hard_block_after_nonpositive_repeats':2}
def task():return {'assignment_id':'T'}

def test_all_provider_blocked_never_reenabled():
    p=QPolicy(cfg(),1); st=EpisodeState.initial(task(),5); d=p.select('T',st.policy_keys(),.3,ACTIONS,[],{},set(ACTIONS)); assert d['action'] is None and d['mode']=='no_available_action'

def test_repetition_block_does_not_remove_only_provider_valid_action():
    p=QPolicy(cfg(),1); st=EpisodeState.initial(task(),5); valid='REFINE_SCOPE'; blocked=set(ACTIONS)-{valid}; recent=[{'action':valid,'reward':-1,'success':False},{'action':valid,'reward':-1,'success':False}]; d=p.select('T',st.policy_keys(),.3,ACTIONS,recent,{},blocked); assert d['action']==valid

def test_q_update_and_combination():
    p=QPolicy(cfg(),1); st=EpisodeState.initial(task(),5); k=st.policy_keys(); p.update('T',k,'REFINE_SCOPE',1.0,k,True); q,vals,visits,active=p.combined('T',k,'REFINE_SCOPE'); assert visits['global']==1 and visits['task']==1 and q>0 and set(active)=={'global','task'}

def test_freeze_disables_epsilon():
    p=QPolicy(cfg(),1); p.frozen=True; st=EpisodeState.initial(task(),5); d=p.select('T',st.policy_keys(),.3,ACTIONS); assert d['effective_epsilon']==0.0
