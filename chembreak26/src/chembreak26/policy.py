from __future__ import annotations
import json, random
from pathlib import Path
from typing import Any
from .constants import ACTIONS
from .utils import stable_hex, utc_now, write_json

class QPolicy:
    """Two-level Q policy for CB26: global state/action knowledge + task-local state/action knowledge.

    The uploaded prompt set has only one functional and one semantic category, so CB26 deliberately
    removes the prior HC/HD/OT pseudo-hierarchy instead of pretending those layers contain information.
    """
    SCHEMA_VERSION=1
    def __init__(self,settings:dict[str,Any],seed:int,data:dict|None=None):
        self.settings=settings; self.seed=int(seed); data=data or {}
        schema=int(data.get('policy_schema_version',self.SCHEMA_VERSION if not data else 0))
        if data and schema!=self.SCHEMA_VERSION:
            raise RuntimeError(f'CB26 policy schema mismatch: expected {self.SCHEMA_VERSION}, found {schema}. Use fresh CB26 storage.')
        self.q_global=dict(data.get('q_global',{})); self.q_task=dict(data.get('q_task',{}))
        self.visits_global=dict(data.get('visits_global',{})); self.visits_task=dict(data.get('visits_task',{}))
        self.decisions=int(data.get('decisions',0)); self.updates=int(data.get('updates',0)); self.frozen=bool(data.get('frozen',False)); self.metadata=dict(data.get('metadata',{}))
    @classmethod
    def load(cls,path,settings,seed):
        p=Path(path); return cls(settings,seed,json.loads(p.read_text()) if p.exists() else None)
    @staticmethod
    def _nested(table,*keys):
        cur=table
        for k in keys:cur=cur.get(str(k),{})
        return cur
    @staticmethod
    def _get(table,keys,action,default=0.0):
        return float(QPolicy._nested(table,*keys).get(action,default))
    @staticmethod
    def _getv(table,keys,action):
        return int(QPolicy._nested(table,*keys).get(action,0))
    @staticmethod
    def _set(table,keys,action,value):
        cur=table
        for k in keys:cur=cur.setdefault(str(k),{})
        cur[action]=value
    def _components(self,task_id,keys,action):
        qg=self._get(self.q_global,[keys['global']],action); qt=self._get(self.q_task,[task_id,keys['task']],action)
        vg=self._getv(self.visits_global,[keys['global']],action); vt=self._getv(self.visits_task,[task_id,keys['task']],action)
        return qg,qt,vg,vt
    def combined(self,task_id,keys,action):
        qg,qt,vg,vt=self._components(task_id,keys,action)
        target=max(1,int(self.settings.get('support_confidence_target_visits',3)))
        cg=min(1.0,vg/target); ct=min(1.0,vt/target)
        wg=float(self.settings.get('global_weight',.55)); wt=float(self.settings.get('task_weight',.45))
        active=[]; num=0.0; den=0.0
        if vg>0 and wg>0:active.append('global'); num+=wg*cg*qg; den+=wg
        if vt>0 and wt>0:active.append('task'); num+=wt*ct*qt; den+=wt
        return (num/den if den else 0.0),{'global':qg,'task':qt},{'global':vg,'task':vt},active
    def _total_visits(self,task_id,keys,a):
        _,_,vg,vt=self._components(task_id,keys,a); return vg+vt
    @staticmethod
    def _trailing_nonpositive(recent,action):
        n=0
        for x in reversed(recent or []):
            if str(x.get('action'))!=action or float(x.get('reward',0))>0 or bool(x.get('success',False)):break
            n+=1
        return n
    def _state_support(self,task_id,keys,actions):
        return sum(self._total_visits(task_id,keys,a) for a in actions)
    def _effective_epsilon(self,task_id,keys,base,actions,recent):
        if self.frozen:return 0.0
        e=float(base)
        if self._state_support(task_id,keys,actions)==0:e+=float(self.settings.get('novel_state_epsilon_bonus',0.0))
        if recent and float(recent[-1].get('reward',0))<=0:e+=float(self.settings.get('negative_feedback_epsilon_bonus',0.0))
        return min(float(self.settings.get('max_effective_epsilon',.35)),max(0.0,e))
    def select(self,task_id,keys,epsilon,actions=None,recent=None,route_bonus=None,extra_blocked=None):
        actions=list(actions or ACTIONS); recent=list(recent or []); route_bonus=dict(route_bonus or {}); blocked=set(extra_blocked or [])
        hard=max(1,int(self.settings.get('hard_block_after_nonpositive_repeats',2)))
        if recent:
            last=str(recent[-1].get('action',''))
            # Repetition blocking is soft relative to provider blocks: if it removes the last valid
            # provider-allowed action, keep that last valid action rather than re-enable provider blocks.
            provider_valid=[a for a in actions if a not in blocked]
            if last in provider_valid and self._trailing_nonpositive(recent,last)>=hard and len(provider_valid)>1:
                blocked.add(last)
        candidates=[a for a in actions if a not in blocked]
        if not candidates:
            return {'action':None,'mode':'no_available_action','base_epsilon':float(epsilon),'effective_epsilon':0.0,'blocked_actions':sorted(blocked),'combined_q':0.0,'q_global':0.0,'q_task':0.0,'route_bonus':0.0,'repeat_penalty':0.0,'state_support_visits':0,'active_components':[]}
        idx=self.decisions+1 if not self.frozen else 0
        if not self.frozen:self.decisions=idx
        eff=self._effective_epsilon(task_id,keys,epsilon,actions,recent)
        details={a:self.combined(task_id,keys,a) for a in candidates}
        penalty={a:float(self.settings.get('repeat_nonpositive_penalty',.75))*self._trailing_nonpositive(recent,a) for a in candidates}
        adjusted={a:details[a][0]+float(route_bonus.get(a,0.0))-penalty[a] for a in candidates}
        supported=[a for a in candidates if details[a][3] or float(route_bonus.get(a,0.0))>0]
        rng=random.Random(self.seed+idx*7919)
        explore=(not self.frozen) and rng.random()<eff
        if explore:
            mv=min(self._total_visits(task_id,keys,a) for a in candidates); pool=[a for a in candidates if self._total_visits(task_id,keys,a)==mv]; action=min(pool,key=lambda a:stable_hex(self.seed,idx,task_id,keys['global'],a)); mode='exploration'
        elif not supported:
            mv=min(self._total_visits(task_id,keys,a) for a in candidates); pool=[a for a in candidates if self._total_visits(task_id,keys,a)==mv]; action=min(pool,key=lambda a:stable_hex(self.seed,task_id,keys['global'],a)); mode='cold_start'
        else:
            best=max(adjusted[a] for a in supported); pool=[a for a in supported if abs(adjusted[a]-best)<1e-12]; action=min(pool,key=lambda a:(self._total_visits(task_id,keys,a),stable_hex(self.seed,task_id,keys['global'],a))); mode='exploitation'
        q,vals,visits,active=details[action]
        return {'action':action,'mode':mode,'base_epsilon':float(epsilon),'effective_epsilon':float(eff),'blocked_actions':sorted(blocked),'combined_q':float(q),'q_global':float(vals['global']),'q_task':float(vals['task']),'route_bonus':float(route_bonus.get(action,0.0)),'repeat_penalty':float(penalty[action]),'state_support_visits':int(sum(visits.values())),'active_components':active,'global_key':keys['global'],'task_key':keys['task']}
    def _update_component(self,qtab,vtab,keys,action,reward,next_keys,terminal,alpha):
        old=self._get(qtab,keys,action); nxt=0.0 if terminal else max(self._get(qtab,next_keys,a) for a in ACTIONS); target=float(reward) if terminal else float(reward)+float(self.settings['discount'])*nxt; new=old+float(alpha)*(target-old); self._set(qtab,keys,action,new); self._set(vtab,keys,action,self._getv(vtab,keys,action)+1); return old,new
    def update(self,task_id,keys,action,reward,next_keys,terminal):
        if self.frozen:raise RuntimeError('Frozen CB26 policy cannot be updated')
        g=self._update_component(self.q_global,self.visits_global,[keys['global']],action,reward,[next_keys['global']],terminal,float(self.settings.get('global_learning_rate',.30)))
        t=self._update_component(self.q_task,self.visits_task,[task_id,keys['task']],action,reward,[task_id,next_keys['task']],terminal,float(self.settings.get('task_learning_rate',.18)))
        self.updates+=1
        return {'old_q_global':g[0],'new_q_global':g[1],'old_q_task':t[0],'new_q_task':t[1]}
    def to_dict(self):
        return {'namespace':'CB26','policy_schema_version':self.SCHEMA_VERSION,'seed':self.seed,'metadata':self.metadata,'q_global':self.q_global,'q_task':self.q_task,'visits_global':self.visits_global,'visits_task':self.visits_task,'decisions':self.decisions,'updates':self.updates,'frozen':self.frozen,'saved_at_utc':utc_now()}
    def save(self,path):write_json(path,self.to_dict())
    def freeze(self,path):self.frozen=True; self.save(path); return self.to_dict()
    def summary(self):return {'global_states':len(self.q_global),'task_memories':len(self.q_task),'decisions':self.decisions,'updates':self.updates,'frozen':self.frozen}
