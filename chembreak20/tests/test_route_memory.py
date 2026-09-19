from chembreak20.route_memory import RouteMemory

def settings():return {'wilson_z':1.96,'target_support_attempts':3,'reliability_weight':.5,'support_weight':.15,'reward_weight':.15,'progress_weight':.2,'reward_scale':4,'guidance_weight':.4,'guidance_top_k':3}
def test_route_stores_actions_only():
    r=RouteMemory(settings()); x=r.observe('T',['REFINE_SCOPE','REPHRASE_GOAL'],success=True,cumulative_reward=5,peak_progress=1,epoch=1,turns=2); assert x['actions']==['REFINE_SCOPE','REPHRASE_GOAL']; assert 'prompt' not in x
def test_route_accumulates_success_failure():
    r=RouteMemory(settings()); a=['REFINE_SCOPE']; r.observe('T',a,success=True,cumulative_reward=5,peak_progress=1,epoch=1,turns=1); r.observe('T',a,success=False,cumulative_reward=-1,peak_progress=.2,epoch=2,turns=4); x=r.ranked('T')[0]; assert x['attempts']==2 and x['successes']==1 and x['failures']==1
def test_route_guidance_matches_prefix():
    r=RouteMemory(settings()); r.observe('T',['REFINE_SCOPE','REPHRASE_GOAL'],success=True,cumulative_reward=5,peak_progress=1,epoch=1,turns=2); b=r.action_bonus('T',['REFINE_SCOPE']); assert b['REPHRASE_GOAL']>0
def test_route_guidance_no_prefix_no_bonus():
    r=RouteMemory(settings()); r.observe('T',['REFINE_SCOPE','REPHRASE_GOAL'],success=True,cumulative_reward=5,peak_progress=1,epoch=1,turns=2); b=r.action_bonus('T',['ALTERNATE_ANGLE']); assert max(b.values())==0
def test_terminal_routes_distinct():
    r=RouteMemory(settings()); r.observe('T',['REFINE_SCOPE'],success=True,cumulative_reward=5,peak_progress=1,epoch=1,turns=1); r.observe('T',['REPHRASE_GOAL'],success=True,cumulative_reward=4,peak_progress=1,epoch=1,turns=1); out=r.terminal_routes('T',2); assert len(out)==2 and out[0]['actions']!=out[1]['actions']
def test_freeze_blocks_update(tmp_path):
    r=RouteMemory(settings()); r.observe('T',['REFINE_SCOPE'],success=True,cumulative_reward=5,peak_progress=1,epoch=1,turns=1); r.freeze(tmp_path/'r.json');
    import pytest
    with pytest.raises(RuntimeError):r.observe('T',['REPHRASE_GOAL'],success=True,cumulative_reward=5,peak_progress=1,epoch=2,turns=1)
