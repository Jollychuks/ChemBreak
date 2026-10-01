from __future__ import annotations
import gc,hashlib,os
from pathlib import Path
from .checkpoint import Store
from .config import load_config,validate_config
from .constants import ACTIONS,NAMESPACE,PACKAGE_VERSION,SOURCE_PROMPTS_SHA256,MANIFEST_SHA256,RESPONSE_CLASS_PROGRESS
from .dataset import selected_tasks
from .metrics import export_results
from .policy import QPolicy
from .prompts import attack_prompt,candidate_judge_prompt,response_judge_prompt
from .providers import ChcsJudgeCascadeError,ProviderPolicyBlock,make_roles
from .reporting import Reporter
from .state import EpisodeState
from .targets import make_target
from .utils import write_json


class ChemBreak28Runner:
    def __init__(self,config_path,target_id):
        self.config=load_config(config_path); validate_config(self.config); self.target_id=str(target_id)
        targets={str(x['id']):x for x in self.config['targets']}
        if self.target_id not in targets:raise ValueError(f'Unknown target {self.target_id}; valid={sorted(targets)}')
        self.target_cfg=targets[self.target_id]; r=self.config['run']; e=self.config['experiment']
        self.tasks=selected_tasks(r['prompts_path'],r['manifest_path']).to_dict('records'); lim=r.get('task_limit'); self.tasks=self.tasks[:int(lim)] if lim else self.tasks
        self.out=Path(r['output_root'])/r['experiment_revision']/self.target_id; self.out.mkdir(parents=True,exist_ok=True); self.store=Store(self.out/'state.sqlite3')
        self.art=Path(r['artifact_root'])/r['experiment_revision']/self.target_id; self.art.mkdir(parents=True,exist_ok=True)
        self.active_task_id=None; self.policy=None; self.task_memory=None; self.loaded=False
        a=self.config['roles']; self.identity={'namespace':NAMESPACE,'package_version':PACKAGE_VERSION,'experiment_revision':r['experiment_revision'],'target_id':self.target_id,'target_model':self.target_cfg['model'],'target_revision':self.target_cfg['revision'],'source_prompts_sha256':SOURCE_PROMPTS_SHA256,'manifest_sha256':MANIFEST_SHA256,'assignment_ids':[str(x['assignment_id']) for x in self.tasks],'seed':int(r['seed']),'models':{'attack_llm':a['attack_llm']['model'],'intent_gate_llm':a['intent_gate_llm']['model'],'chcs_judge_llm':a['chcs_judge_llm']['model'],'chcs_fallback_judge_llm':a['chcs_fallback_judge_llm']['model'],'target':self.target_cfg['model']},'experiment':dict(e),'candidate_gate':dict(self.config['candidate_gate']),'chcs':dict(self.config['chcs']),'policy':dict(self.config['policy']),'reward':dict(self.config['reward']),'technical_failures':dict(self.config['technical_failures']),'task_isolation':{'cross_task_learning':False,'reset_before_each_task':True}}
        existing=self.store.get_meta('experiment_identity')
        if existing is not None and existing!=self.identity:raise RuntimeError('Existing checkpoint belongs to another ChemBreak28 experiment identity. Use a new experiment revision or storage directory.')
        self.store.set_meta('experiment_identity',self.identity); self.store.set_meta('task_count',len(self.tasks))
        active=self.store.get_meta('active_episode_checkpoint')
        if active:
            phase,episode,aid=str(active['phase']),int(active['episode']),str(active['assignment_id'])
            if self.store.episode_status(phase,episode,aid)!='complete':
                self.store.delete_episode_and_turns(phase,episode,aid); self.store.set_meta('task_policy_snapshot',active.get('policy')); self.store.set_meta('task_memory_snapshot',active.get('memory')); self.store.set_meta('task_controller_assignment_id',aid)
            self.store.set_meta('active_episode_checkpoint',None)
        self.store.clear_uncommitted_episodes()
        project_id=None if r['dry_run'] else os.environ.get('GOOGLE_CLOUD_PROJECT'); self.roles=make_roles(self.config,project_id); self.target=make_target(self.target_cfg,r['dry_run'])
        self.reporter=Reporter((1+int(e['adaptive_episodes']))*len(self.tasks),enabled=r.get('live_progress',True))

    def _task_seed(self,aid):return int(hashlib.sha256(f"{self.config['run']['seed']}|{self.target_id}|{aid}".encode()).hexdigest()[:8],16)
    def _task_dir(self,aid):p=self.art/str(aid); p.mkdir(parents=True,exist_ok=True); return p

    def _empty_memory(self,aid):return {'assignment_id':str(aid),'baseline':None,'attempts':[],'decision_history':[],'latest_evaluated':None,'peak_chcs':1,'unresolved_judgments':0}

    def _rebuild_memory(self,aid):
        memory=self._empty_memory(aid); turns=[x for x in self.store.turns() if str(x['assignment_id'])==str(aid)]
        import json
        for row in sorted(turns,key=lambda x:int(x.get('target_query_index',0))):
            judge=json.loads(row.get('judge_json') or '{}'); unresolved=bool(judge.get('judge_unresolved'))
            attempt={'phase':row['phase'],'episode':int(row['episode']),'turn':int(row['turn_index']),'action':row['action_id'],'judge_status':'unresolved' if unresolved else 'judged','chcs':judge.get('chcs'),'response_class':judge.get('response_class','unknown' if unresolved else ''),'reward':float(row.get('reward',0)),'success':bool(judge.get('final_success',False)),'target_response_excerpt':str(row.get('response',''))[-1200:]}
            if row['phase']=='baseline':memory['baseline']=attempt
            else:memory['attempts'].append(attempt)
            if unresolved:memory['unresolved_judgments']+=1
            else:
                memory['latest_evaluated']={'chcs':int(judge['chcs']),'response_class':str(judge['response_class']),'action':row['action_id'],'reward':float(row.get('reward',0))}; memory['peak_chcs']=max(memory['peak_chcs'],int(judge['chcs']))
            try:decision=json.loads(row.get('decision_json') or '{}'); item=decision.get('memory_item');
            except Exception:item=None
            if item:memory['decision_history'].append(item)
        return memory

    def _activate(self,aid):
        aid=str(aid)
        if self.active_task_id==aid:return
        checkpoint_task=self.store.get_meta('task_controller_assignment_id'); seed=self._task_seed(aid)
        ps=self.store.get_meta('task_policy_snapshot') if checkpoint_task==aid else None; ms=self.store.get_meta('task_memory_snapshot') if checkpoint_task==aid else None
        self.policy=QPolicy(self.config['policy'],seed,ps); self.policy.bind_task(aid); self.policy.metadata=self.identity|{'assignment_id':aid,'scope':'task_local'}
        self.task_memory=ms or self._rebuild_memory(aid); self.active_task_id=aid; self.store.set_meta('task_controller_assignment_id',aid); self.store.set_meta('task_policy_snapshot',self.policy.to_dict()); self.store.set_meta('task_memory_snapshot',self.task_memory)
        write_json(self._task_dir(aid)/'task_isolation_manifest.json',{'assignment_id':aid,'target_id':self.target_id,'task_seed':seed,'policy_scope':'current_task_only','memory_scope':'current_task_only','fresh_target_context_each_episode':True,'cross_task_learning':False})

    def _deactivate(self,aid,outcome):
        td=self._task_dir(aid); self.policy.save(td/'final_task_policy.json'); write_json(td/'task_memory_INTERNAL.json',self.task_memory); write_json(td/'task_outcome.json',outcome)
        self.store.set_meta('task_controller_assignment_id',None); self.store.set_meta('task_policy_snapshot',None); self.store.set_meta('task_memory_snapshot',None)
        self.active_task_id=None; self.policy=None; self.task_memory=None

    def load_target(self):
        if not self.loaded:self.target.load(); self.loaded=True
    def unload_target(self):
        if self.loaded:self.target.unload(); self.loaded=False; gc.collect()

    def _gate(self,task,candidate):
        g=self.roles.judge_candidate(candidate_judge_prompt(task,candidate),candidate); c=self.config['candidate_gate']; g['gate_pass']=bool(not g['drifted'] and float(g['goal_preservation'])>=float(c['goal_preservation_min']) and float(g['task_fidelity'])>=float(c['task_fidelity_min'])); return g
    def _judge(self,task,response):
        j=self.roles.judge_response(response_judge_prompt(task,response),response); j['harmfulness_normalized']=(int(j['chcs'])-1)/4.0; j['final_success']=int(j['chcs'])==int(self.config['chcs']['success_score']); return j
    def _reward(self,prev_chcs,j,prev_class):
        r=self.config['reward']; chcs=int(j['chcs']); value=float(r['chcs_delta'])*((chcs-int(prev_chcs))/4.0); value+=float(r['behavioral_progress'])*((RESPONSE_CLASS_PROGRESS.get(str(j['response_class']),0)-RESPONSE_CLASS_PROGRESS.get(str(prev_class),0))/5.0)
        if j['final_success']:value+=float(r['success_bonus'])
        if chcs==1:value-=float(r['chcs1_penalty'])
        if chcs<=int(prev_chcs) and RESPONSE_CLASS_PROGRESS.get(str(j['response_class']),0)<=RESPONSE_CLASS_PROGRESS.get(str(prev_class),0) and not j['final_success']:value-=float(r['stagnation_penalty'])
        return round(value-float(r['turn_cost']),6)

    def _candidate(self,phase,episode,index,task,state,action,last_response,blocked,rejected):
        aid=str(task['assignment_id']); max_reg=max(1,int(self.config['candidate_gate']['max_regenerations_per_action']))
        for regen in range(max_reg):
            try:data=self.roles.attack(attack_prompt(task,action,state,last_response,self.task_memory),action); candidate=str(data['utterance']).strip(); reason=str(data.get('reason_code',''))
            except ProviderPolicyBlock as exc:
                blocked.add(action); self.store.save_provider_event(phase,episode,aid,stage='attack_llm',provider=exc.provider,event_type='policy_block',error_code=exc.code,action_id=action,message=exc.provider_message); return None,None,'',rejected,'attack_policy_block'
            except Exception as exc:
                blocked.add(action); self.store.save_provider_event(phase,episode,aid,stage='attack_llm',provider='vertex_genai',event_type='attack_llm_error',error_code=type(exc).__name__,action_id=action,message=str(exc)); return None,None,'',rejected,'attack_error'
            try:gate=self._gate(task,candidate)
            except Exception as exc:
                blocked.add(action); self.store.save_provider_event(phase,episode,aid,stage='intent_gate',provider='vertex_genai',event_type='gate_error',error_code=type(exc).__name__,action_id=action,message=str(exc)); return None,None,reason,rejected,'gate_error'
            if gate['gate_pass']:return candidate,gate,reason,rejected,'ok'
            rejected+=1; self.store.save_provider_event(phase,episode,aid,stage='candidate_gate',provider='vertex_genai',event_type='goal_drift_rejection',error_code=str(gate.get('reason_code','drift')),action_id=action,details={'goal_preservation':gate['goal_preservation'],'task_fidelity':gate['task_fidelity'],'regeneration':regen+1})
            if rejected>=int(self.config['candidate_gate']['max_rejected_candidates_per_episode']):return None,None,reason,rejected,'rejection_limit'
        blocked.add(action); return None,None,'',rejected,'action_rejected'

    def _target_and_judge(self,phase,episode,task,state,context_id,action,candidate):
        aid=str(task['assignment_id']); turn=state.turn_index+1; ch=hashlib.sha256(candidate.encode()).hexdigest()[:16]
        try:gen=self.target.generate(candidate,state.history); self.reporter.target_queries+=1
        except Exception as exc:
            self.store.save_provider_event(phase,episode,aid,stage='target',provider=self.target_id,event_type='target_error',error_code=type(exc).__name__,action_id=action,message=str(exc)); return None,None,None,'target_error'
        qidx=self.store.save_target_query(phase=phase,episode=episode,assignment_id=aid,context_id=context_id,turn_index=turn,action_id=action,candidate_hash=ch,prompt=candidate,response=gen.text,latency_seconds=gen.latency_seconds)
        try:j=self._judge(task,gen.text); self.store.finalize_target_query(qidx,j,'judged'); return gen,qidx,j,'judged'
        except ChcsJudgeCascadeError as exc:
            self.store.mark_target_query_judge_error(qidx,str(exc),exc.audit); self.store.save_provider_event(phase,episode,aid,stage='chcs_judge_cascade',provider='primary_then_fallback',event_type='unresolved_judgment',error_code='both_judges_failed',action_id=action,details={'target_query_index':qidx,'judge_provenance':exc.audit}); return gen,qidx,{'judge_unresolved':True,'judge_provenance':exc.audit,'final_success':False,'response_class':'unknown','chcs':None},'unresolved'

    def _save_turn(self,phase,episode,task,state,context_id,action,candidate,gate,gen,qidx,judge,reward,decision,reason):
        aid=str(task['assignment_id']); decision=dict(decision); decision['attack_reason_code']=reason
        self.store.save_turn_with_learning(self.policy.to_dict(),self.task_memory,phase=phase,episode=episode,assignment_id=aid,context_id=context_id,turn_index=state.turn_index,target_query_index=qidx,action_id=action,selection_mode=decision.get('mode',''),state_key=decision.get('task_key',''),q_task=decision.get('q_task',0),candidate_hash=hashlib.sha256(candidate.encode()).hexdigest()[:16],candidate_source='attack_llm',candidate_gate=gate,prompt=candidate,response=gen.text,judge=judge,decision=decision,reward=reward,latency_seconds=gen.latency_seconds)

    def run_baseline(self,task,index):
        aid=str(task['assignment_id'])
        if self.store.episode_done('baseline',0,aid):return bool(self.store.episode_row('baseline',0,aid)['success'])
        self.reporter.start_task('baseline',0,index,len(self.tasks),aid); self.store.start_episode('baseline',0,aid)
        state=EpisodeState.initial(task,0,1); candidate=str(task['original_prompt']); gen,qidx,j,status=self._target_and_judge('baseline',0,task,state,f'{aid}:baseline','ORIGINAL_TASK',candidate)
        if gen is None:
            self.store.complete_episode('baseline',0,aid,False,0,0.0,'target_error'); return False
        if status=='judged':state.advance('ORIGINAL_TASK',candidate,gen.text,j,0.0); reason='success' if j['final_success'] else 'baseline_complete'
        else:state.advance_unresolved('ORIGINAL_TASK',candidate,gen.text); reason='baseline_unresolved'
        self.store.save_turn(phase='baseline',episode=0,assignment_id=aid,context_id=f'{aid}:baseline',turn_index=1,target_query_index=qidx,action_id='ORIGINAL_TASK',selection_mode='direct',state_key='baseline',q_task=0,candidate_hash=hashlib.sha256(candidate.encode()).hexdigest()[:16],candidate_source='original_task',candidate_gate={'gate_pass':True,'not_applicable':True},prompt=candidate,response=gen.text,judge=j,decision={'mode':'direct_baseline','learning_update_skipped':True},reward=0.0,latency_seconds=gen.latency_seconds)
        success=bool(j.get('final_success',False)); self.store.complete_episode('baseline',0,aid,success,1,0.0,reason); self.reporter.complete('baseline',0,index,len(self.tasks),aid,success,1,0.0); return success

    def run_episode(self,task,index,episode):
        aid=str(task['assignment_id'])
        if self.store.episode_done('adaptive',episode,aid):return bool(self.store.episode_row('adaptive',episode,aid)['success'])
        self._activate(aid); epsilon=float(self.config['experiment']['episode_epsilons'][episode-1]); max_turns=int(self.config['experiment']['max_turns_per_episode'])
        state=EpisodeState.initial(task,episode,max_turns,self.task_memory); context_id=f'{aid}:episode:{episode}'; blocked=set(); rejected=0; last_response=''; total=0.0; success=False; completion_reason='turn_budget_exhausted'
        self.reporter.start_task('adaptive',episode,index,len(self.tasks),aid); self.store.set_meta('active_episode_checkpoint',{'phase':'adaptive','episode':episode,'assignment_id':aid,'policy':self.policy.to_dict(),'memory':self.task_memory}); self.store.start_episode('adaptive',episode,aid)
        while state.turn_index<max_turns and not success:
            keys=state.policy_keys(); decision=self.policy.select(aid,keys,epsilon,recent=state.decision_history,extra_blocked=blocked); action=decision['action']
            if action is None:completion_reason='no_available_action'; break
            candidate,gate,reason,rejected,status=self._candidate('adaptive',episode,index,task,state,action,last_response,blocked,rejected)
            if candidate is None:
                if status=='rejection_limit':completion_reason=status; break
                continue
            gen,qidx,j,jstatus=self._target_and_judge('adaptive',episode,task,state,context_id,action,candidate)
            if gen is None:blocked.add(action); continue
            if jstatus=='judged':
                prev_chcs=state.chcs; prev_class=state.response_class; reward=self._reward(prev_chcs,j,prev_class); memory_item=state.advance(action,candidate,gen.text,j,reward); success=bool(j['final_success']); update=self.policy.update(aid,keys,action,reward,state.policy_keys(),success or state.turn_index>=max_turns); decision['update']=update
            else:
                reward=float(self.config['technical_failures']['unresolved_reward']); memory_item=state.advance_unresolved(action,candidate,gen.text,self.config['technical_failures']['unresolved_response_class']); decision['update']={'learning_update_skipped':True,'reason':'unresolved_judgment'}; self.task_memory['unresolved_judgments']+=1
            total+=reward; attempt={'phase':'adaptive','episode':episode,'turn':state.turn_index,'action':action,'judge_status':jstatus,'chcs':j.get('chcs'),'response_class':j.get('response_class'),'reward':reward,'success':success,'target_response_excerpt':gen.text[-1200:]}; self.task_memory['attempts'].append(attempt); self.task_memory['decision_history'].append(memory_item)
            if jstatus=='judged':self.task_memory['latest_evaluated']={'chcs':int(j['chcs']),'response_class':str(j['response_class']),'action':action,'reward':reward}; self.task_memory['peak_chcs']=max(int(self.task_memory['peak_chcs']),int(j['chcs']))
            decision['memory_item']=memory_item; self._save_turn('adaptive',episode,task,state,context_id,action,candidate,gate,gen,qidx,j,reward,decision,reason); self.reporter.turn('adaptive',episode,index,len(self.tasks),aid,state.turn_index,action,j,reward,decision,'attack_llm',len(blocked)); last_response=gen.text
        if success:completion_reason='first_validated_success'
        self.store.complete_episode_with_learning('adaptive',episode,aid,success,state.turn_index,total,completion_reason,self.policy.to_dict(),self.task_memory); self.store.set_meta('active_episode_checkpoint',None)
        td=self._task_dir(aid); self.policy.save(td/f'policy_after_episode_{episode}.json'); write_json(td/f'memory_after_episode_{episode}_INTERNAL.json',self.task_memory); self.reporter.complete('adaptive',episode,index,len(self.tasks),aid,success,state.turn_index,total); return success

    def run_all(self):
        self.load_target(); max_episodes=int(self.config['experiment']['adaptive_episodes'])
        try:
            for index,task in enumerate(self.tasks,1):
                aid=str(task['assignment_id']); baseline_success=self.run_baseline(task,index)
                if baseline_success:
                    write_json(self._task_dir(aid)/'task_outcome.json',{'assignment_id':aid,'success':True,'success_phase':'baseline','success_episode':0}); continue
                self._activate(aid); success=False; success_episode=None
                for episode in range(1,max_episodes+1):
                    if self.run_episode(task,index,episode):success=True; success_episode=episode; break
                self._deactivate(aid,{'assignment_id':aid,'success':success,'success_phase':'adaptive' if success else 'none','success_episode':success_episode,'episodes_attempted':success_episode or max_episodes})
            return export_results(self.store,self.out/'release',self.target_id,self.config)
        finally:self.unload_target()

    def close(self):self.store.close()
