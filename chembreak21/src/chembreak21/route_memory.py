from __future__ import annotations
import json, math, hashlib
from pathlib import Path
from typing import Any
from .constants import ACTIONS
from .utils import utc_now, write_json

class RouteMemory:
    """Task-specific exact successful-trajectory memory with reliability evidence.

    CB21 intentionally stores exact attacker prompts for successful/replayed paths so that later
    epochs can test reproducibility in a fresh target conversation.  Abstract actions are stored
    alongside the prompts so the controller can still reason over strategy structure.
    """
    SCHEMA_VERSION=2
    def __init__(self,settings:dict[str,Any],data:dict|None=None):
        self.settings=settings; data=data or {}
        schema=int(data.get('route_schema_version',self.SCHEMA_VERSION if not data else 0))
        if data and schema!=self.SCHEMA_VERSION:raise RuntimeError(f'CB21 route schema mismatch: expected {self.SCHEMA_VERSION}, found {schema}. Use fresh CB21 storage.')
        self.tasks=dict(data.get('tasks',{})); self.metadata=dict(data.get('metadata',{})); self.frozen=bool(data.get('frozen',False)); self.updates=int(data.get('updates',0))
    @classmethod
    def load(cls,path,settings):
        p=Path(path); return cls(settings,json.loads(p.read_text()) if p.exists() else None)
    @staticmethod
    def _step_hash(step):
        return hashlib.sha256((str(step.get('action',''))+'\n'+str(step.get('prompt',''))).encode()).hexdigest()[:20]
    @classmethod
    def route_id(cls,steps):
        material='|'.join(cls._step_hash(s) for s in steps)
        return hashlib.sha256(('cb21-route|'+material).encode()).hexdigest()[:24]
    @staticmethod
    def _clean_steps(steps):
        out=[]
        for s in steps:
            a=str(s.get('action',''))
            p=str(s.get('prompt','')).strip()
            if a not in ACTIONS or not p:continue
            out.append({'action':a,'prompt':p,'prompt_hash':hashlib.sha256(p.encode()).hexdigest()[:20],'candidate_gate':dict(s.get('candidate_gate') or {}),'source':str(s.get('source',''))})
        return out
    def _record(self,task_id,steps):
        steps=self._clean_steps(steps); rid=self.route_id(steps); task=self.tasks.setdefault(str(task_id),{})
        return task.setdefault(rid,{'route_id':rid,'steps':steps,'actions':[s['action'] for s in steps],'attempts':0,'successes':0,'failures':0,'reward_sum':0.0,'peak_quality_sum':0.0,'best_quality':0.0,'turns_to_success':[],'epochs_seen':{},'success_epochs':{},'failure_epochs':{},'created_at_utc':utc_now()})
    def observe(self,task_id,steps,*,success,cumulative_reward,peak_quality,epoch,turns):
        if self.frozen:raise RuntimeError('Frozen CB21 route memory cannot be updated')
        steps=self._clean_steps(steps)
        if not steps:return None
        r=self._record(task_id,steps); e=str(int(epoch)); r['attempts']+=1; r['successes']+=int(bool(success)); r['failures']+=int(not bool(success)); r['reward_sum']+=float(cumulative_reward); r['peak_quality_sum']+=float(peak_quality); r['best_quality']=max(float(r.get('best_quality',0)),float(peak_quality)); r['epochs_seen'][e]=int(r['epochs_seen'].get(e,0))+1; r['last_success']=bool(success); r['last_seen_utc']=utc_now()
        if success:r['success_epochs'][e]=int(r['success_epochs'].get(e,0))+1; r['turns_to_success'].append(int(turns))
        else:r['failure_epochs'][e]=int(r['failure_epochs'].get(e,0))+1
        self.updates+=1; return r
    def observe_existing(self,task_id,route_id,*,success,cumulative_reward,peak_quality,epoch,turns):
        if self.frozen:raise RuntimeError('Frozen CB21 route memory cannot be updated')
        r=self.tasks.get(str(task_id),{}).get(str(route_id))
        if r is None:raise KeyError(f'Unknown route {route_id} for task {task_id}')
        e=str(int(epoch)); r['attempts']+=1; r['successes']+=int(bool(success)); r['failures']+=int(not bool(success)); r['reward_sum']+=float(cumulative_reward); r['peak_quality_sum']+=float(peak_quality); r['best_quality']=max(float(r.get('best_quality',0)),float(peak_quality)); r['epochs_seen'][e]=int(r['epochs_seen'].get(e,0))+1; r['last_success']=bool(success); r['last_seen_utc']=utc_now()
        if success:r['success_epochs'][e]=int(r['success_epochs'].get(e,0))+1; r['turns_to_success'].append(int(turns))
        else:r['failure_epochs'][e]=int(r['failure_epochs'].get(e,0))+1
        self.updates+=1; return r
    def _wilson(self,s,n):
        if n<=0:return 0.0
        z=float(self.settings.get('wilson_z',1.96)); p=s/n; d=1+z*z/n
        return max(0.0,(p+z*z/(2*n)-z*math.sqrt((p*(1-p)+z*z/(4*n))/n))/d)
    def score(self,r):
        n=int(r.get('attempts',0)); s=int(r.get('successes',0)); support=min(1.0,n/max(1,int(self.settings.get('target_support_attempts',3)))); reliability=self._wilson(s,n); mean_reward=float(r.get('reward_sum',0))/n if n else 0.0; reward_norm=(math.tanh(mean_reward/max(.01,float(self.settings.get('reward_scale',4.0))))+1)/2; mean_quality=float(r.get('peak_quality_sum',0))/n if n else 0.0
        return float(self.settings.get('reliability_weight',.55))*reliability + float(self.settings.get('support_weight',.15))*support + float(self.settings.get('reward_weight',.10))*reward_norm + float(self.settings.get('quality_weight',.20))*mean_quality
    def _decorate(self,r):
        x=json.loads(json.dumps(r)); n=max(1,int(x.get('attempts',0))); x['rank_score']=self.score(r); x['success_rate']=float(x.get('successes',0))/n; x['mean_reward']=float(x.get('reward_sum',0))/n; x['mean_peak_quality']=float(x.get('peak_quality_sum',0))/n; x['discovered_success']=bool(int(x.get('successes',0))>=1); x['confirmed_success']=bool(len(x.get('success_epochs',{}))>=2); return x
    def ranked(self,task_id,successful_only=False):
        rows=[]
        for r in self.tasks.get(str(task_id),{}).values():
            if successful_only and int(r.get('successes',0))<=0:continue
            rows.append(self._decorate(r))
        rows.sort(key=lambda x:(-int(x['confirmed_success']),-int(x['discovered_success']),-x['rank_score'],-int(x['successes']),-int(x['attempts']),x['route_id']))
        return rows
    def best_replay_route(self,task_id):
        rows=self.ranked(task_id,successful_only=True); return rows[0] if rows else None
    def final_routes(self,task_id,k=2):
        return self.ranked(task_id,successful_only=True)[:max(0,int(k))]
    def action_bonus(self,task_id,prefix):
        prefix=list(prefix); bonus={a:0.0 for a in ACTIONS}; weight=float(self.settings.get('guidance_weight',.35)); topk=max(1,int(self.settings.get('guidance_top_k',3)))
        for r in self.ranked(task_id,successful_only=True)[:topk]:
            acts=list(r['actions'])
            if len(acts)>len(prefix) and acts[:len(prefix)]==prefix:
                nxt=acts[len(prefix)]; bonus[nxt]=max(bonus[nxt],weight*float(r['rank_score']))
        return bonus
    def coverage(self):
        tasks=len(self.tasks); success=sum(1 for t in self.tasks.values() if any(int(r.get('successes',0))>0 for r in t.values())); confirmed=sum(1 for t in self.tasks.values() if any(len(r.get('success_epochs',{}))>=2 for r in t.values())); routes=sum(len(t) for t in self.tasks.values())
        return {'tasks_with_routes':tasks,'tasks_with_successful_routes':success,'tasks_with_confirmed_routes':confirmed,'routes':routes,'updates':self.updates}
    def public_rankings(self):
        out={}
        for tid in self.tasks:
            out[tid]=[]
            for r in self.ranked(tid):
                out[tid].append({k:r[k] for k in ('route_id','actions','attempts','successes','failures','rank_score','success_rate','discovered_success','confirmed_success','success_epochs','failure_epochs','turns_to_success')})
        return out
    def to_dict(self):return {'namespace':'CB21','route_schema_version':self.SCHEMA_VERSION,'metadata':self.metadata,'tasks':self.tasks,'updates':self.updates,'frozen':self.frozen,'saved_at_utc':utc_now()}
    def save(self,path):write_json(path,self.to_dict())
    def freeze(self,path):self.frozen=True; self.save(path); return self.to_dict()
