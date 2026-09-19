from pathlib import Path
import yaml,json
from chembreak21.runner import ChemBreak21Runner
from chembreak21.state import EpisodeState

ROOT=Path(__file__).resolve().parents[1]

def temp_config(tmp_path,limit=1):
    c=yaml.safe_load((ROOT/'configs/config.cb21.yaml').read_text()); c['run']['dry_run']=True; c['run']['task_limit']=limit; c['run']['live_progress']=False; c['run']['output_root']=str(tmp_path/'runs'); c['run']['artifact_root']=str(tmp_path/'policies'); c['run']['prompts_path']=str(ROOT/'data/prompts.csv'); c['run']['manifest_path']=str(ROOT/'data/CB21_prompts28_manifest_v1.csv'); c['run']['lock_path']=str(ROOT/'data/CB21_prompts28_lock_v1.json'); p=tmp_path/'config.yaml'; p.write_text(yaml.safe_dump(c,sort_keys=False)); return p

class ScriptedRoles:
    def __init__(self):self.n=0
    def attack(self,prompt,action):self.n+=1; return {'action_id':action,'reason_code':'test','utterance':f'fresh-{self.n}-{action}'}
    def judge_candidate(self,prompt,candidate):return {'goal_preservation':1.0,'task_fidelity':1.0,'drifted':False,'confidence':1.0,'reason_code':'same'}
    def judge_response(self,prompt,response):
        success='SUCCESS' in response
        return {'response_class':'substantive_compliance' if success else 'partial_compliance','goal_progress':1.0 if success else .2,'task_fidelity':1.0,'response_relevance':1.0,'response_correctness':1.0,'chemistry_relevance':1.0,'policy_violation':2 if success else 0,'actionable_harm':2 if success else 0,'confidence':1.0,'reason_code':'test'}

class ReplayRecoveryTarget:
    def __init__(self):self.prompts=[]; self.histories=[]
    def load(self):pass
    def unload(self):pass
    def generate(self,prompt,history):
        from chembreak21.targets import TargetResponse
        self.prompts.append(prompt); self.histories.append(list(history))
        # Stored route p1,p2 fails; second recovery candidate succeeds.
        return TargetResponse('SUCCESS' if str(prompt).startswith('fresh-2-') else 'not yet',.001)

class TerminalTarget:
    def __init__(self):self.calls=[]
    def load(self):pass
    def unload(self):pass
    def generate(self,prompt,history):
        from chembreak21.targets import TargetResponse
        self.calls.append((prompt,len(history)))
        # First route a1,a2 fails; second route b1,b2 succeeds at b2.
        return TargetResponse('SUCCESS' if prompt=='b2' else 'not yet',.001)

def test_dry_run_small_end_to_end(tmp_path):
    r=ChemBreak21Runner(temp_config(tmp_path,2),'ChemDFM')
    try:
        s=r.run_all(); assert s['baseline']['episodes']==2; assert s['learning_epoch_1']['episodes']==2; assert s['learning_epoch_2']['episodes']==2; assert s['learning_epoch_3']['episodes']==2; assert s['terminal']['episodes']==2
    finally:r.close()

def test_epoch2_replays_two_turn_route_then_uses_two_recovery_turns(tmp_path):
    r=ChemBreak21Runner(temp_config(tmp_path,1),'ChemDFM')
    try:
        task=r.tasks[0]; aid=task['assignment_id']; gate={'goal_preservation':1.0,'task_fidelity':1.0,'drifted':False,'confidence':1.0,'reason_code':'x','gate_pass':True}; route=r.routes.observe(aid,[{'action':'REFINE_SCOPE','prompt':'p1','candidate_gate':gate},{'action':'REPHRASE_GOAL','prompt':'p2','candidate_gate':gate}],success=True,cumulative_reward=5,peak_quality=1,epoch=1,turns=2)
        r.roles=ScriptedRoles(); r.target=ReplayRecoveryTarget(); r.loaded=True; r.store.start_episode('learning',2,aid)
        s,t,_=r._learning_episode(task,2,.2,1)
        assert s is True and t==4
        assert r.target.prompts[:2]==['p1','p2']
        assert r.target.prompts[2].startswith('fresh-1-') and r.target.prompts[3].startswith('fresh-2-')
        rr=r.routes.ranked(aid); old=[x for x in rr if x['route_id']==route['route_id']][0]; assert old['failures']>=1
        assert any(x['successes']>=1 and len(x['steps'])==4 for x in rr if x['route_id']!=route['route_id'])
    finally:r.close()

def test_four_turn_replay_failure_has_no_recovery_turn(tmp_path):
    r=ChemBreak21Runner(temp_config(tmp_path,1),'ChemDFM')
    try:
        task=r.tasks[0]; aid=task['assignment_id']; gate={'goal_preservation':1.0,'task_fidelity':1.0,'drifted':False,'confidence':1.0,'reason_code':'x','gate_pass':True}; steps=[{'action':'REFINE_SCOPE','prompt':f'p{i}','candidate_gate':gate} for i in range(1,5)]; r.routes.observe(aid,steps,success=True,cumulative_reward=5,peak_quality=1,epoch=1,turns=4); roles=ScriptedRoles(); r.roles=roles; r.target=ReplayRecoveryTarget(); r.loaded=True; r.target.generate=lambda prompt,history: __import__('chembreak21.targets',fromlist=['TargetResponse']).TargetResponse('not yet',.001); r.store.start_episode('learning',2,aid); s,t,_=r._learning_episode(task,2,.2,1); assert s is False and t==4 and roles.n==0
    finally:r.close()

def test_terminal_executes_full_first_route_then_resets_and_full_second_route(tmp_path):
    r=ChemBreak21Runner(temp_config(tmp_path,1),'ChemDFM')
    try:
        task=r.tasks[0]; aid=task['assignment_id']; gate={'goal_preservation':1.0,'task_fidelity':1.0,'drifted':False,'confidence':1.0,'reason_code':'x','gate_pass':True}; r.routes.observe(aid,[{'action':'REFINE_SCOPE','prompt':'a1','candidate_gate':gate},{'action':'REPHRASE_GOAL','prompt':'a2','candidate_gate':gate}],success=True,cumulative_reward=5,peak_quality=1,epoch=1,turns=2); r.routes.observe(aid,[{'action':'CHANGE_REPRESENTATION','prompt':'b1','candidate_gate':gate},{'action':'ALTERNATE_ANGLE','prompt':'b2','candidate_gate':gate}],success=True,cumulative_reward=4,peak_quality=.9,epoch=2,turns=2); r.roles=ScriptedRoles(); r.target=TerminalTarget(); r.loaded=True; r.policy.frozen=True; r.routes.frozen=True; s,t,_,reason=r._terminal_episode(task,1); assert s is True and t==4; assert [x[0] for x in r.target.calls]==['a1','a2','b1','b2']; assert [x[1] for x in r.target.calls]==[0,2,0,2]; assert reason=='success_route_2'
    finally:r.close()

def test_baseline_judge_error_preserves_raw_target_query(tmp_path):
    class BadJudgeRoles(ScriptedRoles):
        def judge_response(self,prompt,response):raise RuntimeError('judge down')
    class OneTarget:
        def load(self):pass
        def unload(self):pass
        def generate(self,prompt,history):
            from chembreak21.targets import TargetResponse
            return TargetResponse('raw response',.001)
    r=ChemBreak21Runner(temp_config(tmp_path,1),'ChemDFM')
    try:r.roles=BadJudgeRoles(); r.target=OneTarget(); r.loaded=True; r.run_baseline(); q=r.store.target_queries(); assert len(q)==1 and q[0]['response']=='raw response' and q[0]['judge_status']=='judge_error'; ep=r.store.episodes()[0]; assert ep['terminal_reason']=='judge_error_after_target'
    finally:r.close()

def test_targets_have_separate_directories(tmp_path):
    p=temp_config(tmp_path,1); a=ChemBreak21Runner(p,'ChemDFM'); b=ChemBreak21Runner(p,'ChemLLM')
    try:assert a.out!=b.out and a.art!=b.art
    finally:a.close(); b.close()
