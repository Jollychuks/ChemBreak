from .test_dry_run import config
from chembreak18.runner import ChemBreak18Runner
from chembreak18.state import EpisodeState


class FreshRoles:
    def attack(self, prompt, action):
        return {'action_id': action, 'reason': 'fresh', 'utterance': f'fresh::{action}'}
    def judge(self, prompt, response):
        raise AssertionError('judge is not used in this unit test')


def test_learning_replay_does_not_reintroduce_repetition_blocked_action(tmp_path):
    r=ChemBreak18Runner(config(tmp_path,1))
    try:
        task=r.tasks[0]; aid=str(task['assignment_id']); state=EpisodeState.initial(task,4)
        keys=state.policy_keys()
        r.evidence.record(task_id=aid,prompt='remembered success',action_id='REFINE_SCOPE',global_key=keys['global'],task_key=keys['task'],phase='learning',epoch=1,turn_index=1,success=True,reward=2.0,combined_q=1.0,source='attack_llm_fresh')
        r.roles=FreshRoles()
        decision=r.policy.describe(aid,keys,'DECOMPOSE_GOAL',0.0,recent=state.decision_history,mode='exploitation')
        decision['blocked_actions']=['REFINE_SCOPE']
        action,prompt,_,source,_=r._learning_candidate(task,state,0.0,decision,set(),'')
        assert action=='DECOMPOSE_GOAL'
        assert prompt=='fresh::DECOMPOSE_GOAL'
        assert source=='attack_llm_fresh'
    finally:
        r.close()
