from __future__ import annotations
import json, math, hashlib
from pathlib import Path
from typing import Any
from .constants import ACTIONS
from .utils import utc_now, write_json

class RouteMemory:
    """Task-specific exact successful-trajectory memory with replay evidence.

    CB23 stores exact attacker prompts only in internal state so later epochs can test whether a
    previously successful path replicates in a fresh target context. Public exports contain only
    hashes/actions/statistics. Route ranking is lexicographic and is driven by observed CHCS=5
    success reliability rather than chemical-correctness or relevance scores.
    """
    SCHEMA_VERSION=3

    def __init__(self,settings:dict[str,Any],data:dict|None=None):
        self.settings=settings; data=data or {}
        schema=int(data.get('route_schema_version',self.SCHEMA_VERSION if not data else 0))
        if data and schema!=self.SCHEMA_VERSION:
            raise RuntimeError(f'CB23 route schema mismatch: expected {self.SCHEMA_VERSION}, found {schema}. Use fresh CB23 storage.')
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
        return hashlib.sha256(('cb23-route|'+material).encode()).hexdigest()[:24]

    @staticmethod
    def _clean_steps(steps):
        out=[]
        for s in steps:
            a=str(s.get('action','')); p=str(s.get('prompt','')).strip()
            if a not in ACTIONS or not p:continue
            out.append({
                'action':a,
                'prompt':p,
                'prompt_hash':hashlib.sha256(p.encode()).hexdigest()[:20],
                'candidate_gate':dict(s.get('candidate_gate') or {}),
                'source':str(s.get('source','')),
            })
        return out

    def _record(self,task_id,steps):
        steps=self._clean_steps(steps); rid=self.route_id(steps); task=self.tasks.setdefault(str(task_id),{})
        return task.setdefault(rid,{
            'route_id':rid,
            'steps':steps,
            'actions':[s['action'] for s in steps],
            'attempts':0,'successes':0,'failures':0,
            'reward_sum':0.0,'peak_chcs_sum':0.0,'terminal_chcs_sum':0.0,'best_peak_chcs':1,
            'turns_to_success':[],
            'epochs_seen':{},'success_epochs':{},'failure_epochs':{},
            'created_at_utc':utc_now(),
        })

    def _observe_record(self,r,*,success,cumulative_reward,peak_chcs,terminal_chcs,epoch,turns):
        e=str(int(epoch)); r['attempts']+=1; r['successes']+=int(bool(success)); r['failures']+=int(not bool(success))
        r['reward_sum']+=float(cumulative_reward); r['peak_chcs_sum']+=float(peak_chcs); r['terminal_chcs_sum']+=float(terminal_chcs)
        r['best_peak_chcs']=max(int(r.get('best_peak_chcs',1)),int(peak_chcs)); r['epochs_seen'][e]=int(r['epochs_seen'].get(e,0))+1
        r['last_success']=bool(success); r['last_seen_utc']=utc_now()
        if success:
            r['success_epochs'][e]=int(r['success_epochs'].get(e,0))+1; r['turns_to_success'].append(int(turns))
        else:
            r['failure_epochs'][e]=int(r['failure_epochs'].get(e,0))+1
        self.updates+=1; return r

    def observe(self,task_id,steps,*,success,cumulative_reward,peak_chcs,terminal_chcs,epoch,turns):
        if self.frozen:raise RuntimeError('Frozen CB23 route memory cannot be updated')
        steps=self._clean_steps(steps)
        if not steps:return None
        return self._observe_record(self._record(task_id,steps),success=success,cumulative_reward=cumulative_reward,peak_chcs=peak_chcs,terminal_chcs=terminal_chcs,epoch=epoch,turns=turns)

    def observe_existing(self,task_id,route_id,*,success,cumulative_reward,peak_chcs,terminal_chcs,epoch,turns):
        if self.frozen:raise RuntimeError('Frozen CB23 route memory cannot be updated')
        r=self.tasks.get(str(task_id),{}).get(str(route_id))
        if r is None:raise KeyError(f'Unknown route {route_id} for task {task_id}')
        return self._observe_record(r,success=success,cumulative_reward=cumulative_reward,peak_chcs=peak_chcs,terminal_chcs=terminal_chcs,epoch=epoch,turns=turns)

    def _wilson(self,s,n):
        if n<=0:return 0.0
        z=float(self.settings.get('wilson_z',1.96)); p=s/n; d=1+z*z/n
        return max(0.0,(p+z*z/(2*n)-z*math.sqrt((p*(1-p)+z*z/(4*n))/n))/d)

    def _decorate(self,r):
        x=json.loads(json.dumps(r)); n=max(1,int(x.get('attempts',0))); s=int(x.get('successes',0))
        x['success_rate']=float(s)/n; x['wilson_lcb']=self._wilson(s,n); x['rank_score']=x['wilson_lcb']
        x['mean_reward']=float(x.get('reward_sum',0))/n; x['mean_peak_chcs']=float(x.get('peak_chcs_sum',0))/n; x['mean_terminal_chcs']=float(x.get('terminal_chcs_sum',0))/n
        x['mean_turns_to_success']=(sum(map(float,x.get('turns_to_success',[])))/len(x.get('turns_to_success',[]))) if x.get('turns_to_success') else None
        x['discovered_success']=bool(s>=1); x['confirmed_success']=bool(len(x.get('success_epochs',{}))>=2)
        return x

    @staticmethod
    def _rank_key(x):
        # Confirmed routes first, then empirical replications/reliability/CHCS and efficiency.
        mts=float(x['mean_turns_to_success']) if x.get('mean_turns_to_success') is not None else 1e9
        return (
            -int(x['confirmed_success']),
            -int(x['successes']),
            -float(x['wilson_lcb']),
            -float(x['mean_terminal_chcs']),
            -float(x['mean_peak_chcs']),
            mts,
            -float(x['mean_reward']),
            str(x['route_id']),
        )

    def ranked(self,task_id,successful_only=False):
        rows=[]
        for r in self.tasks.get(str(task_id),{}).values():
            if successful_only and int(r.get('successes',0))<=0:continue
            rows.append(self._decorate(r))
        rows.sort(key=self._rank_key); return rows

    def best_replay_route(self,task_id):
        rows=self.ranked(task_id,successful_only=True); return rows[0] if rows else None

    def final_routes(self,task_id,k=2):
        return self.ranked(task_id,successful_only=True)[:max(0,int(k))]

    def action_bonus(self,task_id,prefix):
        prefix=list(prefix); bonus={a:0.0 for a in ACTIONS}; weight=float(self.settings.get('guidance_weight',.35)); topk=max(1,int(self.settings.get('guidance_top_k',3)))
        for r in self.ranked(task_id,successful_only=True)[:topk]:
            acts=list(r['actions'])
            if len(acts)>len(prefix) and acts[:len(prefix)]==prefix:
                nxt=acts[len(prefix)]
                reliability=float(r['wilson_lcb'])
                confirmed_bonus=0.15 if r['confirmed_success'] else 0.0
                bonus[nxt]=max(bonus[nxt],weight*min(1.0,reliability+confirmed_bonus))
        return bonus

    def coverage(self):
        tasks=len(self.tasks); success=sum(1 for t in self.tasks.values() if any(int(r.get('successes',0))>0 for r in t.values())); confirmed=sum(1 for t in self.tasks.values() if any(len(r.get('success_epochs',{}))>=2 for r in t.values())); routes=sum(len(t) for t in self.tasks.values())
        return {'tasks_with_routes':tasks,'tasks_with_successful_routes':success,'tasks_with_confirmed_routes':confirmed,'routes':routes,'updates':self.updates}

    def public_rankings(self):
        out={}
        fields=('route_id','actions','attempts','successes','failures','rank_score','success_rate','wilson_lcb','mean_peak_chcs','mean_terminal_chcs','mean_turns_to_success','mean_reward','discovered_success','confirmed_success','success_epochs','failure_epochs','turns_to_success')
        for tid in self.tasks:
            out[tid]=[{k:r[k] for k in fields} for r in self.ranked(tid)]
        return out

    def to_dict(self):return {'namespace':'CB23','route_schema_version':self.SCHEMA_VERSION,'metadata':self.metadata,'tasks':self.tasks,'updates':self.updates,'frozen':self.frozen,'saved_at_utc':utc_now()}
    def save(self,path):write_json(path,self.to_dict())
    def freeze(self,path):self.frozen=True; self.save(path); return self.to_dict()
