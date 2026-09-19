from pathlib import Path
import yaml,json
from chembreak20.runner import ChemBreak20Runner

ROOT=Path(__file__).resolve().parents[1]

def temp_config(tmp_path,limit=2):
    c=yaml.safe_load((ROOT/'configs/config.cb20.yaml').read_text()); c['run']['dry_run']=True; c['run']['task_limit']=limit; c['run']['live_progress']=False; c['run']['output_root']=str(tmp_path/'runs'); c['run']['artifact_root']=str(tmp_path/'policies'); c['run']['task_bank_path']=str(ROOT/'data/final_task_bank.csv'); c['run']['mini_manifest_path']=str(ROOT/'data/CB20_mini24_manifest_v1.csv'); c['run']['mini_lock_path']=str(ROOT/'data/CB20_mini24_lock_v1.json'); p=tmp_path/'config.yaml'; p.write_text(yaml.safe_dump(c,sort_keys=False)); return p

def test_two_targets_use_separate_directories(tmp_path):
    p=temp_config(tmp_path,1); a=ChemBreak20Runner(p,'ChemDFM'); b=ChemBreak20Runner(p,'ChemLLM');
    try:assert a.out!=b.out and a.art!=b.art
    finally:a.close(); b.close()
def test_dry_run_full_small(tmp_path):
    p=temp_config(tmp_path,2); r=ChemBreak20Runner(p,'ChemDFM')
    try:
        s=r.run_all(); assert s['baseline']['episodes']==2; assert s['learning_epoch_1']['episodes']==2; assert s['learning_epoch_2']['episodes']==2; assert s['learning_epoch_3']['episodes']==2; assert s['terminal']['episodes']==2; assert r.policy.updates==24; assert r.routes.updates==6
    finally:r.close()
def test_resume_does_not_duplicate(tmp_path):
    p=temp_config(tmp_path,1); r=ChemBreak20Runner(p,'ChemDFM')
    try:r.run_all(); u=r.policy.updates; ru=r.routes.updates
    finally:r.close()
    r2=ChemBreak20Runner(p,'ChemDFM')
    try:r2.run_all(); assert r2.policy.updates==u and r2.routes.updates==ru
    finally:r2.close()
def test_freeze_snapshot_exists(tmp_path):
    p=temp_config(tmp_path,1); r=ChemBreak20Runner(p,'ChemDFM')
    try:r.run_baseline(); r.run_learning(); r.freeze(); assert r.freeze_snapshot_path.exists() and r.policy.frozen and r.routes.frozen
    finally:r.close()
def test_learning_orders_recorded(tmp_path):
    p=temp_config(tmp_path,3); r=ChemBreak20Runner(p,'ChemDFM')
    try:r.run_baseline(); r.run_learning(); assert r.store.get_meta('epoch_1_assignment_order') and r.store.get_meta('epoch_2_assignment_order')
    finally:r.close()

class _AttackErrorRoles:
    def attack(self,prompt,action):
        raise RuntimeError('simulated attack service failure')
    def judge_candidate(self,prompt,candidate):
        raise AssertionError('candidate judge must not run after attack failure')
    def judge_response(self,prompt,response):
        raise AssertionError('response judge must not run after attack failure')

def test_attack_llm_error_is_logged_and_action_skipped(tmp_path):
    from chembreak20.state import EpisodeState
    p=temp_config(tmp_path,1); r=ChemBreak20Runner(p,'ChemDFM')
    try:
        task=r.tasks[0]; state=EpisodeState.initial(task,4); r.roles=_AttackErrorRoles(); blocked=set()
        candidate,gate,reason,rejects,status=r._candidate_for_action('learning',1,1,task,state,'REFINE_SCOPE','',{},None,[],blocked,0)
        assert candidate is None and gate is None and status=='attack_llm_error'
        assert 'REFINE_SCOPE' in blocked
        events=r.store.provider_events(); assert events[-1]['event_type']=='attack_llm_error'
    finally:r.close()

class _FailingTarget:
    def load(self):return None
    def unload(self):return None
    def generate(self,prompt,history):raise RuntimeError('simulated target failure')

def test_baseline_target_error_does_not_crash_run(tmp_path):
    p=temp_config(tmp_path,1); r=ChemBreak20Runner(p,'ChemDFM')
    try:
        r.target=_FailingTarget(); r.loaded=True; s=r.run_baseline()
        assert s['baseline']['episodes']==1 and s['baseline']['successes']==0
        ep=r.store.episodes()[0]; assert ep['turns']==0 and ep['terminal_reason']=='target_error'
        assert any(x['event_type']=='target_error' for x in r.store.provider_events())
    finally:r.close()

def test_partial_learning_episode_restores_pre_episode_state(tmp_path):
    from chembreak20.state import EpisodeState
    p=temp_config(tmp_path,1); r=ChemBreak20Runner(p,'ChemDFM')
    task=r.tasks[0]; aid=str(task['assignment_id']); st=EpisodeState.initial(task,4); keys=st.policy_keys(); pre_p=r.policy.to_dict(); pre_r=r.routes.to_dict()
    r.store.set_meta('active_episode_checkpoint',{'phase':'learning','epoch':1,'assignment_id':aid,'policy':pre_p,'routes':pre_r}); r.store.start_episode('learning',1,aid)
    r.policy.update(aid,keys,'REFINE_SCOPE',1.0,keys,True); r.routes.observe(aid,['REFINE_SCOPE'],success=True,cumulative_reward=1,peak_progress=1,epoch=1,turns=1)
    r.store.save_turn_with_learning(r.policy.to_dict(),r.routes.to_dict(),phase='learning',epoch=1,assignment_id=aid,turn_index=1,action_id='REFINE_SCOPE',selection_mode='exploration',state_key=keys['global'],route_id='',route_rank=None,q_global=0,q_task=0,combined_q=0,route_bonus=0,candidate_hash='x',candidate_source='test',candidate_gate={},prompt='x',response='y',judge={},decision={},reward=1,latency_seconds=0)
    r.close()
    r2=ChemBreak20Runner(p,'ChemDFM')
    try:
        assert r2.policy.updates==int(pre_p['updates'])
        assert r2.routes.updates==int(pre_r['updates'])
        assert r2.store.episode_status('learning',1,aid) is None
        assert [x for x in r2.store.turns() if x['phase']=='learning' and x['assignment_id']==aid]==[]
    finally:r2.close()

class _TerminalRoles:
    def __init__(self):self.n=0
    def attack(self,prompt,action):
        self.n+=1; return {'action_id':action,'reason_code':'test','utterance':f'test-candidate-{self.n}-{action}'}
    def judge_candidate(self,prompt,candidate):return {'goal_preservation':1.0,'task_fidelity':1.0,'drifted':False,'confidence':1.0,'reason_code':'same_goal'}
    def judge_response(self,prompt,response):return {'response_class':'refusal','goal_progress':0.0,'task_fidelity':1.0,'response_relevance':0.5,'response_correctness':0.5,'chemistry_relevance':0.5,'policy_violation':0,'actionable_harm':0,'confidence':1.0,'reason_code':'no'}

def test_terminal_uses_two_distinct_learned_routes_before_synthesis(tmp_path):
    p=temp_config(tmp_path,1); r=ChemBreak20Runner(p,'ChemDFM')
    try:
        aid=str(r.tasks[0]['assignment_id'])
        r.routes.observe(aid,['REFINE_SCOPE'],success=True,cumulative_reward=5,peak_progress=1,epoch=1,turns=1)
        r.routes.observe(aid,['REPHRASE_GOAL'],success=True,cumulative_reward=4,peak_progress=.9,epoch=2,turns=1)
        r.roles=_TerminalRoles(); r.target.load(); r.loaded=True
        u0=r.policy.updates; ru0=r.routes.updates
        success,turns,_,_=r._terminal_episode(r.tasks[0],1)
        rows=[x for x in r.store.turns() if x['phase']=='terminal']
        assert success is False and turns==4
        assert [x['candidate_source'] for x in rows[:2]]==['learned_route','learned_route']
        assert rows[0]['route_id'] and rows[1]['route_id'] and rows[0]['route_id']!=rows[1]['route_id']
        assert rows[2]['candidate_source']=='synthesized_fallback' and rows[3]['candidate_source']=='synthesized_fallback'
        assert r.policy.updates==u0 and r.routes.updates==ru0
    finally:r.close()
