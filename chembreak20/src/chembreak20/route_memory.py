from __future__ import annotations
import json, math
from pathlib import Path
from typing import Any
from .constants import ACTIONS
from .utils import stable_hex,utc_now,write_json

class RouteMemory:
    """Task-specific memory over abstract action sequences only.

    Exact generated prompts are never replayed.  The route memory tracks which
    sequences of abstract actions were attempted, how often they succeeded, and
    how much progress/reward they produced.  The Attack LLM re-realizes a route
    from the immutable source task in each new episode.
    """
    SCHEMA_VERSION=1
    def __init__(self,settings:dict[str,Any],data:dict|None=None):
        self.settings=settings; data=data or {}
        schema=int(data.get('route_schema_version',self.SCHEMA_VERSION if not data else 0))
        if data and schema!=self.SCHEMA_VERSION:raise RuntimeError(f'CB20 route schema mismatch: expected {self.SCHEMA_VERSION}, found {schema}')
        self.tasks=dict(data.get('tasks',{})); self.metadata=dict(data.get('metadata',{})); self.frozen=bool(data.get('frozen',False)); self.updates=int(data.get('updates',0))
    @classmethod
    def load(cls,path,settings):
        p=Path(path); return cls(settings,json.loads(p.read_text()) if p.exists() else None)
    @staticmethod
    def route_id(actions):return stable_hex('route',*actions,length=20)
    def _record(self,task_id,actions):
        task=self.tasks.setdefault(str(task_id),{})
        rid=self.route_id(actions)
        return task.setdefault(rid,{'route_id':rid,'actions':list(actions),'attempts':0,'successes':0,'failures':0,'reward_sum':0.0,'peak_progress_sum':0.0,'best_progress':0.0,'turns_to_success':[],'epochs':{},'last_success':False,'created_at_utc':utc_now()})
    def observe(self,task_id,actions,*,success,cumulative_reward,peak_progress,epoch,turns):
        if self.frozen:raise RuntimeError('Frozen CB20 route memory cannot be updated')
        actions=[str(a) for a in actions if str(a) in ACTIONS]
        if not actions:return None
        r=self._record(task_id,actions); r['attempts']+=1; r['successes']+=int(bool(success)); r['failures']+=int(not bool(success)); r['reward_sum']+=float(cumulative_reward); r['peak_progress_sum']+=float(peak_progress); r['best_progress']=max(float(r.get('best_progress',0)),float(peak_progress)); r['last_success']=bool(success); r['last_seen_utc']=utc_now(); r['epochs'][str(int(epoch))]=int(r['epochs'].get(str(int(epoch)),0))+1
        if success:r['turns_to_success'].append(int(turns))
        self.updates+=1; return r
    def _wilson(self,s,n):
        if n<=0:return 0.0
        z=float(self.settings.get('wilson_z',1.96)); p=s/n; d=1+z*z/n; return max(0.0,(p+z*z/(2*n)-z*math.sqrt((p*(1-p)+z*z/(4*n))/n))/d)
    def score(self,r):
        n=int(r.get('attempts',0)); s=int(r.get('successes',0)); support=min(1.0,n/max(1,int(self.settings.get('target_support_attempts',3)))); reliability=self._wilson(s,n); mean_reward=float(r.get('reward_sum',0))/n if n else 0.0; reward_norm=(math.tanh(mean_reward/max(.01,float(self.settings.get('reward_scale',4.0))))+1)/2; mean_peak=float(r.get('peak_progress_sum',0))/n if n else 0.0
        return float(self.settings.get('reliability_weight',.50))*reliability + float(self.settings.get('support_weight',.15))*support + float(self.settings.get('reward_weight',.15))*reward_norm + float(self.settings.get('progress_weight',.20))*mean_peak
    def ranked(self,task_id,successful_only=False):
        rows=[]
        for r in self.tasks.get(str(task_id),{}).values():
            if successful_only and int(r.get('successes',0))<=0:continue
            x=dict(r); x['rank_score']=self.score(r); x['success_rate']=float(r.get('successes',0))/max(1,int(r.get('attempts',0))); x['mean_reward']=float(r.get('reward_sum',0))/max(1,int(r.get('attempts',0))); x['mean_peak_progress']=float(r.get('peak_progress_sum',0))/max(1,int(r.get('attempts',0))); rows.append(x)
        rows.sort(key=lambda x:(-int(x['successes']>0),-x['rank_score'],-int(x['attempts']),x['route_id']))
        return rows
    def _matching_routes(self,task_id,prefix):
        prefix=list(prefix)
        out=[]
        for r in self.ranked(task_id):
            acts=list(r['actions'])
            if len(acts)>len(prefix) and acts[:len(prefix)]==prefix:out.append(r)
        return out
    def action_bonus(self,task_id,prefix):
        matches=self._matching_routes(task_id,prefix); bonus={a:0.0 for a in ACTIONS}; weight=float(self.settings.get('guidance_weight',0.40)); topk=max(1,int(self.settings.get('guidance_top_k',3)))
        for r in matches[:topk]:
            nxt=r['actions'][len(prefix)]; bonus[nxt]=max(bonus[nxt],weight*float(r['rank_score']))
        return bonus
    @staticmethod
    def _different(a,b):
        return tuple(a)!=tuple(b)
    def terminal_routes(self,task_id,k=2):
        ranked=self.ranked(task_id)
        selected=[]
        # Prefer empirically successful routes, then best-supported unsuccessful routes.
        ordered=[r for r in ranked if r['successes']>0]+[r for r in ranked if r['successes']==0]
        for r in ordered:
            if all(self._different(r['actions'],x['actions']) for x in selected):selected.append(r)
            if len(selected)>=k:break
        return selected
    def coverage(self):
        task_count=len(self.tasks); with_success=sum(1 for t in self.tasks.values() if any(int(r.get('successes',0))>0 for r in t.values())); routes=sum(len(t) for t in self.tasks.values())
        return {'tasks_with_routes':task_count,'tasks_with_successful_routes':with_success,'routes':routes,'updates':self.updates}
    def to_dict(self):return {'namespace':'CB20','route_schema_version':self.SCHEMA_VERSION,'metadata':self.metadata,'tasks':self.tasks,'updates':self.updates,'frozen':self.frozen,'saved_at_utc':utc_now()}
    def save(self,path):write_json(path,self.to_dict())
    def freeze(self,path):self.frozen=True; self.save(path); return self.to_dict()
