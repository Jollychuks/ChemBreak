from __future__ import annotations
import json,random
from pathlib import Path
from .constants import ACTIONS
from .utils import stable_hex,utc_now,write_json


class QPolicy:
    """Task-local Q policy carried across one task's adaptive episodes only."""
    SCHEMA_VERSION=3

    def __init__(self,settings,seed,data=None):
        self.settings=settings; self.seed=int(seed); data=data or {}
        if data and int(data.get('policy_schema_version',0))!=self.SCHEMA_VERSION: raise RuntimeError('CB28 policy schema mismatch; use fresh CB28 storage')
        self.q=dict(data.get('q',{})); self.visits=dict(data.get('visits',{})); self.stagnation=dict(data.get('stagnation',{}))
        self.decisions=int(data.get('decisions',0)); self.updates=int(data.get('updates',0)); self.assignment_id=str(data.get('assignment_id','')); self.metadata=dict(data.get('metadata',{}))

    @staticmethod
    def _get(table,key,action,default=0.0):return float(table.get(str(key),{}).get(action,default))
    @staticmethod
    def _getv(table,key,action):return int(table.get(str(key),{}).get(action,0))
    @staticmethod
    def _set(table,key,action,value):table.setdefault(str(key),{})[action]=value

    def bind_task(self,task_id):
        task_id=str(task_id)
        if self.assignment_id and self.assignment_id!=task_id: raise RuntimeError(f'CB28 task-isolation violation: {self.assignment_id} -> {task_id}')
        self.assignment_id=task_id

    def _assert(self,task_id):
        if not self.assignment_id:self.assignment_id=str(task_id)
        if self.assignment_id!=str(task_id):raise RuntimeError('CB28 cross-task policy use is forbidden')

    def _effective_epsilon(self,key,base,recent):
        e=float(base)
        if sum(self._getv(self.visits,key,a) for a in ACTIONS)==0:e+=float(self.settings.get('novel_state_epsilon_bonus',0))
        if recent and float(recent[-1].get('reward',0))<=0:e+=float(self.settings.get('negative_feedback_epsilon_bonus',0))
        return min(float(self.settings.get('max_effective_epsilon',.35)),max(0.0,e))

    def select(self,task_id,keys,epsilon,recent=None,extra_blocked=None):
        self._assert(task_id); key=keys['task']; recent=list(recent or []); blocked=set(extra_blocked or [])
        hard=max(1,int(self.settings.get('hard_block_after_nonpositive_repeats',2)))
        stagnant=[a for a in ACTIONS if a not in blocked and int(self.stagnation.get(key,{}).get(a,0))>=hard]
        if len([a for a in ACTIONS if a not in blocked])>len(stagnant):blocked.update(stagnant)
        choices=[a for a in ACTIONS if a not in blocked]
        if not choices:return {'action':None,'mode':'no_available_action','base_epsilon':float(epsilon),'effective_epsilon':0.0,'blocked_actions':sorted(blocked),'combined_q':0.0,'q_task':0.0,'repeat_penalty':0.0,'state_support_visits':0,'task_key':key}
        self.decisions+=1; idx=self.decisions; eff=self._effective_epsilon(key,epsilon,recent)
        penalty={a:float(self.settings.get('repeat_nonpositive_penalty',.75))*int(self.stagnation.get(key,{}).get(a,0)) for a in choices}
        adjusted={a:self._get(self.q,key,a)-penalty[a] for a in choices}; supported=[a for a in choices if self._getv(self.visits,key,a)>0]
        rng=random.Random(self.seed+idx*7919)
        if rng.random()<eff:
            mv=min(self._getv(self.visits,key,a) for a in choices); pool=[a for a in choices if self._getv(self.visits,key,a)==mv]; action=min(pool,key=lambda a:stable_hex(self.seed,idx,task_id,key,a)); mode='exploration'
        elif not supported:
            mv=min(self._getv(self.visits,key,a) for a in choices); pool=[a for a in choices if self._getv(self.visits,key,a)==mv]; action=min(pool,key=lambda a:stable_hex(self.seed,task_id,key,a)); mode='cold_start'
        else:
            best=max(adjusted[a] for a in supported); pool=[a for a in supported if abs(adjusted[a]-best)<1e-12]; action=min(pool,key=lambda a:(self._getv(self.visits,key,a),stable_hex(self.seed,task_id,key,a))); mode='exploitation'
        return {'action':action,'mode':mode,'base_epsilon':float(epsilon),'effective_epsilon':eff,'blocked_actions':sorted(blocked),'combined_q':self._get(self.q,key,action),'q_task':self._get(self.q,key,action),'repeat_penalty':penalty[action],'state_support_visits':self._getv(self.visits,key,action),'task_key':key}

    def update(self,task_id,keys,action,reward,next_keys,done):
        self._assert(task_id); key=keys['task']; next_key=next_keys['task']; old=self._get(self.q,key,action)
        nxt=0.0 if done else max(self._get(self.q,next_key,a) for a in ACTIONS)
        target=float(reward) if done else float(reward)+float(self.settings['discount'])*nxt
        new=old+float(self.settings['learning_rate'])*(target-old); self._set(self.q,key,action,new); self._set(self.visits,key,action,self._getv(self.visits,key,action)+1)
        sc=self.stagnation.setdefault(key,{}); sc[action]=0 if float(reward)>0 or done else int(sc.get(action,0))+1; self.updates+=1
        return {'old_q_task':old,'new_q_task':new,'learning_update_skipped':False}

    def to_dict(self):return {'namespace':'CB28','policy_schema_version':self.SCHEMA_VERSION,'seed':self.seed,'assignment_id':self.assignment_id,'metadata':self.metadata,'q':self.q,'visits':self.visits,'stagnation':self.stagnation,'decisions':self.decisions,'updates':self.updates,'saved_at_utc':utc_now()}
    def save(self,path):write_json(path,self.to_dict())
    @classmethod
    def load(cls,path,settings,seed):
        p=Path(path); return cls(settings,seed,json.loads(p.read_text()) if p.exists() else None)
