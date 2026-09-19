from chembreak21.route_memory import RouteMemory

def cfg():return {'wilson_z':1.96,'target_support_attempts':3,'reliability_weight':.55,'support_weight':.15,'reward_weight':.10,'quality_weight':.20,'reward_scale':4,'guidance_weight':.35,'guidance_top_k':3}
def steps(a='REFINE_SCOPE',p='p1'):return [{'action':a,'prompt':p,'candidate_gate':{'gate_pass':True},'source':'test'}]

def test_exact_prompt_is_stored():
    r=RouteMemory(cfg()); x=r.observe('T',steps(),success=True,cumulative_reward=5,peak_quality=1,epoch=1,turns=1); assert x['steps'][0]['prompt']=='p1'

def test_same_exact_route_accumulates_across_epochs():
    r=RouteMemory(cfg()); x=r.observe('T',steps(),success=True,cumulative_reward=5,peak_quality=1,epoch=1,turns=1); r.observe_existing('T',x['route_id'],success=True,cumulative_reward=5,peak_quality=1,epoch=2,turns=1); y=r.ranked('T')[0]; assert y['successes']==2 and y['confirmed_success'] is True and set(y['success_epochs'])=={'1','2'}

def test_failed_replay_reduces_success_rate_but_does_not_delete():
    r=RouteMemory(cfg()); x=r.observe('T',steps(),success=True,cumulative_reward=5,peak_quality=1,epoch=1,turns=1); r.observe_existing('T',x['route_id'],success=False,cumulative_reward=-1,peak_quality=.1,epoch=2,turns=1); y=r.ranked('T')[0]; assert y['attempts']==2 and y['successes']==1 and y['failures']==1 and y['success_rate']==.5

def test_best_replay_route_is_successful_only():
    r=RouteMemory(cfg()); r.observe('T',steps('REFINE_SCOPE','bad'),success=False,cumulative_reward=10,peak_quality=1,epoch=1,turns=1); good=r.observe('T',steps('REPHRASE_GOAL','good'),success=True,cumulative_reward=1,peak_quality=.8,epoch=1,turns=1); assert r.best_replay_route('T')['route_id']==good['route_id']

def test_final_routes_rank_confirmed_first():
    r=RouteMemory(cfg()); a=r.observe('T',steps('REFINE_SCOPE','a'),success=True,cumulative_reward=5,peak_quality=1,epoch=1,turns=1); b=r.observe('T',steps('REPHRASE_GOAL','b'),success=True,cumulative_reward=5,peak_quality=1,epoch=1,turns=1); r.observe_existing('T',b['route_id'],success=True,cumulative_reward=5,peak_quality=1,epoch=2,turns=1); out=r.final_routes('T',2); assert out[0]['route_id']==b['route_id'] and len(out)==2

def test_public_rankings_exclude_exact_prompt_text():
    r=RouteMemory(cfg()); r.observe('T',steps(),success=True,cumulative_reward=5,peak_quality=1,epoch=1,turns=1); s=str(r.public_rankings()); assert 'p1' not in s and 'steps' not in s
