from __future__ import annotations
import json, random
from pathlib import Path
from typing import Any
from .constants import ACTIONS
from .utils import stable_hex,utc_now,write_json

class HierarchicalQPolicy:
    SCHEMA_VERSION=7
    def __init__(self,settings:dict[str,Any],seed:int,data:dict|None=None):
        self.settings=settings; self.seed=int(seed); data=data or {}
        schema=int(data.get('policy_schema_version',self.SCHEMA_VERSION if not data else 0))
        if data and schema!=self.SCHEMA_VERSION:raise RuntimeError(f'CB20 policy schema mismatch: expected {self.SCHEMA_VERSION}, found {schema}. Use fresh CB20 storage.')
        for name in ('q_global','q_hc','q_hd','q_ot','q_task','visits_global','visits_hc','visits_hd','visits_ot','visits_task'):
            setattr(self,name,data.get(name,{}))
        self.decisions=int(data.get('decisions',0)); self.updates=int(data.get('updates',0)); self.frozen=bool(data.get('frozen',False)); self.metadata=dict(data.get('metadata',{}))
    @classmethod
    def load(cls,path,settings,seed):
        p=Path(path); return cls(settings,seed,json.loads(p.read_text()) if p.exists() else None)
    @staticmethod
    def _nested_q(table,*keys,action):
        cur=table
        for k in keys:cur=cur.get(str(k),{})
        return float(cur.get(action,0.0))
    @staticmethod
    def _nested_v(table,*keys,action):
        cur=table
        for k in keys:cur=cur.get(str(k),{})
        return int(cur.get(action,0))
    @staticmethod
    def _set_nested(table,keys,action,value):
        cur=table
        for k in keys:cur=cur.setdefault(str(k),{})
        cur[action]=value
    def _values(self,task_id,keys,action):
        g,t,hc,hd,ot=keys['global'],keys['task'],keys['hc'],keys['hd'],keys['ot']
        return {'global':self._nested_q(self.q_global,g,action=action),'hc':self._nested_q(self.q_hc,hc,g,action=action),'hd':self._nested_q(self.q_hd,hd,g,action=action),'ot':self._nested_q(self.q_ot,ot,g,action=action),'task':self._nested_q(self.q_task,task_id,t,action=action)}
    def _visits(self,task_id,keys,action):
        g,t,hc,hd,ot=keys['global'],keys['task'],keys['hc'],keys['hd'],keys['ot']
        return {'global':self._nested_v(self.visits_global,g,action=action),'hc':self._nested_v(self.visits_hc,hc,g,action=action),'hd':self._nested_v(self.visits_hd,hd,g,action=action),'ot':self._nested_v(self.visits_ot,ot,g,action=action),'task':self._nested_v(self.visits_task,task_id,t,action=action)}
    def _weights(self):return {k:float(self.settings.get(f'{k}_weight',d)) for k,d in {'global':.45,'hc':.15,'hd':.15,'ot':.15,'task':.10}.items()}
    def _conf(self,visits):
        target=max(1,int(self.settings.get('support_confidence_target_visits',3))); return {k:min(1.0,float(v)/target) for k,v in visits.items()}
    def combined(self,task_id,keys,action):
        vals=self._values(task_id,keys,action); visits=self._visits(task_id,keys,action); weights=self._weights(); active=[k for k in vals if visits[k]>0 and weights[k]>0]
        if not active:return 0.0,vals,visits,[]
        conf=self._conf(visits); denom=sum(weights[k] for k in active); score=sum(weights[k]*conf[k]*vals[k] for k in active)/denom
        return float(score),vals,visits,active
    def _total_visits(self,task_id,keys,a):return sum(self._visits(task_id,keys,a).values())
    def _state_support(self,task_id,keys,actions):return sum(self._total_visits(task_id,keys,a) for a in actions)
    @staticmethod
    def _trailing_nonpositive(recent,action):
        n=0
        for x in reversed(recent):
            if str(x.get('action'))!=action or float(x.get('reward',0))>0 or bool(x.get('success',False)):break
            n+=1
        return n
    def _effective_epsilon(self,task_id,keys,base,actions,recent):
        if self.frozen:return 0.0,{'novel_state_bonus':0.0,'negative_feedback_bonus':0.0}
        novelty=float(self.settings.get('novel_state_epsilon_bonus',0.0)) if self._state_support(task_id,keys,actions)==0 else 0.0
        neg=float(self.settings.get('negative_feedback_epsilon_bonus',0.0)) if recent and float(recent[-1].get('reward',0))<=0 else 0.0
        cap=float(self.settings.get('max_effective_epsilon',1.0)); return min(cap,max(0.0,float(base)+novelty+neg)),{'novel_state_bonus':novelty,'negative_feedback_bonus':neg}
    def _details(self,task_id,keys,action,epsilon,actions,recent,mode,blocked,route_bonus):
        q,vals,visits,active=self.combined(task_id,keys,action); conf=self._conf(visits); penalty=float(self.settings.get('repeat_nonpositive_penalty',0))*self._trailing_nonpositive(recent,action)
        eff,parts=self._effective_epsilon(task_id,keys,epsilon,actions,recent)
        rb=float(route_bonus.get(action,0.0))
        return {'action':action,'mode':mode,'base_epsilon':float(epsilon),'effective_epsilon':float(eff),**parts,'global_key':keys['global'],'task_key':keys['task'],'hc_id':keys['hc'],'hd_id':keys['hd'],'ot_id':keys['ot'],'state_support_visits':int(self._state_support(task_id,keys,actions)),'active_components':active,'q_global':vals['global'],'q_hc':vals['hc'],'q_hd':vals['hd'],'q_ot':vals['ot'],'q_task':vals['task'],'visits_global':visits['global'],'visits_hc':visits['hc'],'visits_hd':visits['hd'],'visits_ot':visits['ot'],'visits_task':visits['task'],'confidence_global':conf['global'],'confidence_hc':conf['hc'],'confidence_hd':conf['hd'],'confidence_ot':conf['ot'],'confidence_task':conf['task'],'combined_q':q,'route_bonus':rb,'repeat_penalty':penalty,'adjusted_score':q+rb-penalty,'blocked_actions':list(blocked)}
    def select(self,task_id,keys,epsilon,actions=None,recent=None,route_bonus=None,extra_blocked=None):
        actions=list(actions or ACTIONS); recent=list(recent or []); route_bonus=dict(route_bonus or {}); extra_blocked=set(extra_blocked or [])
        idx=self.decisions+1 if not self.frozen else 0
        if not self.frozen:self.decisions=idx
        eff,_=self._effective_epsilon(task_id,keys,epsilon,actions,recent)
        blocked=set(extra_blocked); hard=max(1,int(self.settings.get('hard_block_after_nonpositive_repeats',2)))
        if recent:
            last=str(recent[-1].get('action',''))
            if last in actions and self._trailing_nonpositive(recent,last)>=hard and len(actions)>1:blocked.add(last)
        candidates=[a for a in actions if a not in blocked] or list(actions)
        details={a:self.combined(task_id,keys,a) for a in actions}; penalties={a:float(self.settings.get('repeat_nonpositive_penalty',0))*self._trailing_nonpositive(recent,a) for a in actions}; adjusted={a:details[a][0]+float(route_bonus.get(a,0))-penalties[a] for a in actions}
        supported=[a for a in candidates if details[a][3] or float(route_bonus.get(a,0))>0]
        rng=random.Random(self.seed+idx*7919); explore=(not self.frozen) and rng.random()<eff
        if explore:
            mv=min(self._total_visits(task_id,keys,a) for a in candidates); pool=[a for a in candidates if self._total_visits(task_id,keys,a)==mv]; action=min(pool,key=lambda a:stable_hex(self.seed,idx,task_id,keys['global'],a)); mode='exploration'
        elif not supported:
            mv=min(self._total_visits(task_id,keys,a) for a in candidates); pool=[a for a in candidates if self._total_visits(task_id,keys,a)==mv]; action=min(pool,key=lambda a:stable_hex(self.seed,task_id,keys['global'],a)); mode='cold_start'
        else:
            best=max(adjusted[a] for a in supported); pool=[a for a in supported if abs(adjusted[a]-best)<1e-12]; action=min(pool,key=lambda a:(self._total_visits(task_id,keys,a),stable_hex(self.seed,task_id,keys['global'],a))); mode='exploitation'
        return self._details(task_id,keys,action,epsilon,actions,recent,mode,sorted(blocked),route_bonus)
    def _update_component(self,qtab,vtab,keys,action,reward,next_keys,terminal,alpha):
        old=self._nested_q(qtab,*keys,action=action); nxt=0.0 if terminal else max(self._nested_q(qtab,*next_keys,action=a) for a in ACTIONS); target=float(reward) if terminal else float(reward)+float(self.settings['discount'])*nxt; new=old+float(alpha)*(target-old); self._set_nested(qtab,keys,action,new); self._set_nested(vtab,keys,action,self._nested_v(vtab,*keys,action=action)+1); return old,new
    def update(self,task_id,keys,action,reward,next_keys,terminal):
        if self.frozen:raise RuntimeError('Frozen CB20 policy cannot be updated')
        ag=float(self.settings.get('global_learning_rate',.30)); ac=float(self.settings.get('context_learning_rate',.25)); at=float(self.settings.get('task_learning_rate',.15))
        g=self._update_component(self.q_global,self.visits_global,[keys['global']],action,reward,[next_keys['global']],terminal,ag); hc=self._update_component(self.q_hc,self.visits_hc,[keys['hc'],keys['global']],action,reward,[next_keys['hc'],next_keys['global']],terminal,ac); hd=self._update_component(self.q_hd,self.visits_hd,[keys['hd'],keys['global']],action,reward,[next_keys['hd'],next_keys['global']],terminal,ac); ot=self._update_component(self.q_ot,self.visits_ot,[keys['ot'],keys['global']],action,reward,[next_keys['ot'],next_keys['global']],terminal,ac); task=self._update_component(self.q_task,self.visits_task,[task_id,keys['task']],action,reward,[task_id,next_keys['task']],terminal,at); self.updates+=1
        return {'old_q_global':g[0],'new_q_global':g[1],'old_q_task':task[0],'new_q_task':task[1]}
    def to_dict(self):return {'namespace':'CB20','policy_schema_version':self.SCHEMA_VERSION,'seed':self.seed,'metadata':self.metadata,'q_global':self.q_global,'q_hc':self.q_hc,'q_hd':self.q_hd,'q_ot':self.q_ot,'q_task':self.q_task,'visits_global':self.visits_global,'visits_hc':self.visits_hc,'visits_hd':self.visits_hd,'visits_ot':self.visits_ot,'visits_task':self.visits_task,'decisions':self.decisions,'updates':self.updates,'frozen':self.frozen,'saved_at_utc':utc_now()}
    def save(self,path):write_json(path,self.to_dict())
    def freeze(self,path):self.frozen=True; self.save(path); return self.to_dict()
    def summary(self):return {'global_states':len(self.q_global),'hc_groups':len(self.q_hc),'hd_groups':len(self.q_hd),'ot_groups':len(self.q_ot),'task_memories':len(self.q_task),'decisions':self.decisions,'updates':self.updates,'frozen':self.frozen}
QPolicy=HierarchicalQPolicy
