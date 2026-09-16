from .test_dry_run import config
from chembreak18.runner import ChemBreak18Runner
from chembreak18.providers import ProviderPolicyBlock

class BlockOnceRoles:
    def __init__(self):self.attack_calls=0
    def attack(self,prompt,action):
        self.attack_calls+=1
        if self.attack_calls==1:raise ProviderPolicyBlock('openai','bio_policy','blocked')
        return {'action_id':action,'reason':'mock','utterance':f'[MOCK {action}]'}
    def judge(self,prompt,response):
        return {'response_class':'refusal','goal_progress':0.0,'task_fidelity':1.0,'chemistry_relevance':0.2,'policy_violation':0,'actionable_harm':0,'success':False,'confidence':0.99}

def test_policy_block_is_logged_and_learning_continues(tmp_path):
    r=ChemBreak18Runner(config(tmp_path,1))
    try:
        r.roles=BlockOnceRoles(); r.run_baseline(); s=r.run_learning()
        events=r.store.provider_events()
        assert len(events)==1 and events[0]['error_code']=='bio_policy'
        assert s['learning_epoch_1']['episodes']==1 and s['learning_epoch_2']['episodes']==1 and s['learning_epoch_3']['episodes']==1
        assert s['provider_policy_blocks']['events']==1
        assert len(r.store.turns())==1+3*4  # baseline + target-query turns; policy block consumes no target turn
    finally:r.close()

class AlwaysBlockRoles(BlockOnceRoles):
    def attack(self,prompt,action):
        self.attack_calls+=1
        raise ProviderPolicyBlock('openai','bio_policy','blocked')

def test_all_actions_blocked_completes_episode_and_pipeline_moves_on(tmp_path):
    r=ChemBreak18Runner(config(tmp_path,1))
    try:
        roles=AlwaysBlockRoles(); r.roles=roles; r.run_baseline(); s=r.run_learning()
        learning=[x for x in r.store.episodes() if x['phase']=='learning']
        assert len(learning)==3 and all(x['status']=='complete' for x in learning)
        assert all(x['terminal_reason']=='attack_llm_policy_blocked_all_actions' for x in learning)
        assert len(r.store.provider_events())==18  # six abstract actions x three epochs
        assert len(r.store.turns())==1  # baseline only; no blocked provider event is a ChemDFM turn
        assert s['provider_policy_blocks']['events']==18
    finally:r.close()
