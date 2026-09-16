from __future__ import annotations
import json, random
from pathlib import Path
from typing import Any
from .constants import ACTIONS
from .utils import stable_hex, utc_now, write_json

class HierarchicalQPolicy:
    """CB19 hierarchical tabular policy.

    Reusable value is separated into global behavior, HC/HD/OT context, and a
    lightweight task residual.  CB19 adds evidence memory outside this class;
    the Q-policy remains independently inspectable. Early Q evidence is support-shrunk so one lucky observation cannot dominate cross-task exploitation.
    """
    SCHEMA_VERSION = 6

    def __init__(self, settings: dict[str,Any], seed: int, data: dict|None=None):
        self.settings=settings; self.seed=int(seed); data=data or {}
        schema=int(data.get('policy_schema_version', self.SCHEMA_VERSION if not data else 1))
        if data and schema != self.SCHEMA_VERSION:
            raise RuntimeError(f'CB19 policy schema mismatch: expected {self.SCHEMA_VERSION}, found {schema}. Use fresh CB19 policy storage.')
        self.q_global=data.get('q_global',{}); self.q_hc=data.get('q_hc',{}); self.q_hd=data.get('q_hd',{}); self.q_ot=data.get('q_ot',{}); self.q_task=data.get('q_task',{})
        self.visits_global=data.get('visits_global',{}); self.visits_hc=data.get('visits_hc',{}); self.visits_hd=data.get('visits_hd',{}); self.visits_ot=data.get('visits_ot',{}); self.visits_task=data.get('visits_task',{})
        self.decisions=int(data.get('decisions',0)); self.updates=int(data.get('updates',0)); self.frozen=bool(data.get('frozen',False)); self.metadata=dict(data.get('metadata',{}))

    @classmethod
    def load(cls,path,settings,seed):
        p=Path(path); return cls(settings,seed,json.loads(p.read_text()) if p.exists() else None)

    @staticmethod
    def _nested_q(table,*keys,action):
        cur=table
        for k in keys: cur=cur.get(str(k),{})
        return float(cur.get(action,0.0))
    @staticmethod
    def _nested_v(table,*keys,action):
        cur=table
        for k in keys: cur=cur.get(str(k),{})
        return int(cur.get(action,0))
    @staticmethod
    def _set_nested(table,keys,action,value):
        cur=table
        for k in keys: cur=cur.setdefault(str(k),{})
        cur[action]=value

    def _values(self, task_id: str, keys: dict[str,str], action: str):
        g=keys['global']; t=keys['task']; hc=keys['hc']; hd=keys['hd']; ot=keys['ot']
        return {'global':self._nested_q(self.q_global,g,action=action),'hc':self._nested_q(self.q_hc,hc,g,action=action),'hd':self._nested_q(self.q_hd,hd,g,action=action),'ot':self._nested_q(self.q_ot,ot,g,action=action),'task':self._nested_q(self.q_task,task_id,t,action=action)}
    def _visits(self, task_id: str, keys: dict[str,str], action: str):
        g=keys['global']; t=keys['task']; hc=keys['hc']; hd=keys['hd']; ot=keys['ot']
        return {'global':self._nested_v(self.visits_global,g,action=action),'hc':self._nested_v(self.visits_hc,hc,g,action=action),'hd':self._nested_v(self.visits_hd,hd,g,action=action),'ot':self._nested_v(self.visits_ot,ot,g,action=action),'task':self._nested_v(self.visits_task,task_id,t,action=action)}
    def _weights(self):
        return {'global':float(self.settings.get('global_weight',0.45)),'hc':float(self.settings.get('hc_weight',0.15)),'hd':float(self.settings.get('hd_weight',0.15)),'ot':float(self.settings.get('ot_weight',0.15)),'task':float(self.settings.get('task_weight',0.10))}
    def _support_confidences(self, visits: dict[str,int]):
        target=max(1,int(self.settings.get('support_confidence_target_visits',3)))
        return {k:min(1.0,max(0.0,float(v)/float(target))) for k,v in visits.items()}

    def combined(self, task_id: str, keys: dict[str,str], action: str):
        vals=self._values(task_id,keys,action); visits=self._visits(task_id,keys,action); weights=self._weights(); active=[k for k in vals if visits[k]>0 and weights[k]>0]
        if not active:return 0.0,vals,visits,[]
        # CB19 confidence-shrinks each learned component until it has repeated
        # support.  The base hierarchical weights are unchanged; only the value
        # contributed by a one-off observation is attenuated.
        conf=self._support_confidences(visits)
        denom=sum(weights[k] for k in active)
        score=sum(weights[k]*conf[k]*vals[k] for k in active)/denom
        return float(score),vals,visits,active
    def _action_total_visits(self,task_id,keys,action):return sum(self._visits(task_id,keys,action).values())
    def _state_support(self,task_id,keys,actions):return sum(self._action_total_visits(task_id,keys,a) for a in actions)

    @staticmethod
    def _trailing_nonpositive_repeats(recent, action):
        count=0
        for item in reversed(recent):
            if str(item.get('action'))!=action or float(item.get('reward',0.0))>0 or bool(item.get('success',False)):break
            count+=1
        return count

    def _effective_epsilon(self,task_id,keys,base_epsilon,actions,recent):
        if self.frozen:return 0.0,{'novel_state_bonus':0.0,'negative_feedback_bonus':0.0}
        support=self._state_support(task_id,keys,actions)
        novelty=float(self.settings.get('novel_state_epsilon_bonus',0.0)) if support==0 else 0.0
        negative=float(self.settings.get('negative_feedback_epsilon_bonus',0.0)) if recent and float(recent[-1].get('reward',0.0))<=0 else 0.0
        cap=float(self.settings.get('max_effective_epsilon',1.0))
        return min(cap,max(0.0,float(base_epsilon)+novelty+negative)),{'novel_state_bonus':novelty,'negative_feedback_bonus':negative}

    def _payload(self,task_id,keys,action,epsilon,actions,recent,mode,blocked=None):
        actions=list(actions or ACTIONS); recent=list(recent or []); blocked=list(blocked or [])
        eff,eps_parts=self._effective_epsilon(task_id,keys,epsilon,actions,recent)
        repeat_unit=float(self.settings.get('repeat_nonpositive_penalty',0.0)); penalty=repeat_unit*self._trailing_nonpositive_repeats(recent,action)
        combined_q,vals,visits,active=self.combined(task_id,keys,action); conf=self._support_confidences(visits)
        return {'action':action,'mode':mode,'base_epsilon':float(epsilon),'effective_epsilon':float(eff),**eps_parts,'global_key':keys['global'],'task_key':keys['task'],'hc_id':keys['hc'],'hd_id':keys['hd'],'ot_id':keys['ot'],'state_support_visits':int(self._state_support(task_id,keys,actions)),'active_components':active,'q_global':vals['global'],'q_hc':vals['hc'],'q_hd':vals['hd'],'q_ot':vals['ot'],'q_task':vals['task'],'visits_global':visits['global'],'visits_hc':visits['hc'],'visits_hd':visits['hd'],'visits_ot':visits['ot'],'visits_task':visits['task'],'confidence_global':conf['global'],'confidence_hc':conf['hc'],'confidence_hd':conf['hd'],'confidence_ot':conf['ot'],'confidence_task':conf['task'],'combined_q':combined_q,'repeat_penalty':penalty,'adjusted_score':combined_q-penalty,'blocked_actions':blocked}

    def describe(self,task_id,keys,action,epsilon,actions=None,recent=None,mode='external_selection'):
        return self._payload(task_id,keys,action,epsilon,list(actions or ACTIONS),list(recent or []),mode)

    def select(self,task_id,keys,epsilon,actions=None,recent=None):
        actions=list(actions or ACTIONS); recent=list(recent or [])
        decision_index=self.decisions+1 if not self.frozen else 0
        if not self.frozen:self.decisions=decision_index
        eff,_=self._effective_epsilon(task_id,keys,epsilon,actions,recent)
        hard_after=max(1,int(self.settings.get('hard_block_after_nonpositive_repeats',2))); blocked=[]
        if recent:
            last=str(recent[-1].get('action',''))
            if last in actions and self._trailing_nonpositive_repeats(recent,last)>=hard_after and len(actions)>1:blocked=[last]
        candidates=[a for a in actions if a not in blocked] or list(actions)
        repeat_unit=float(self.settings.get('repeat_nonpositive_penalty',0.0)); details={a:self.combined(task_id,keys,a) for a in actions}
        raw={a:details[a][0] for a in actions}; penalties={a:repeat_unit*self._trailing_nonpositive_repeats(recent,a) for a in actions}; adjusted={a:raw[a]-penalties[a] for a in actions}
        supported=[a for a in candidates if len(details[a][3])>0]
        rng=random.Random(self.seed+decision_index*7919)
        explore=(not self.frozen) and (rng.random()<eff)
        if explore:
            # Exploration may deliberately choose an unsupported action.
            minv=min(self._action_total_visits(task_id,keys,a) for a in candidates); pool=[a for a in candidates if self._action_total_visits(task_id,keys,a)==minv]
            action=min(pool,key=lambda a:stable_hex(self.seed,decision_index,task_id,keys['global'],keys['task'],a)); mode='exploration'
        elif not supported:
            # No action has learned evidence in this state.  Do not mislabel
            # an unsupported zero-valued action as exploitation.
            minv=min(self._action_total_visits(task_id,keys,a) for a in candidates); pool=[a for a in candidates if self._action_total_visits(task_id,keys,a)==minv]
            action=min(pool,key=lambda a:stable_hex(self.seed,task_id,keys['global'],keys['task'],a)); mode='cold_start'
        else:
            # Exploitation is restricted to actions with actual learned support
            # in at least one active hierarchical component.
            best=max(adjusted[a] for a in supported); pool=[a for a in supported if abs(adjusted[a]-best)<1e-12]
            action=min(pool,key=lambda a:(self._action_total_visits(task_id,keys,a),stable_hex(self.seed,task_id,keys['global'],keys['task'],a))); mode='exploitation'
        payload=self._payload(task_id,keys,action,epsilon,actions,recent,mode,blocked)
        payload['effective_epsilon']=float(eff); payload['repeat_penalty']=penalties[action]; payload['adjusted_score']=adjusted[action]
        return payload

    def _update_component(self,qtable,vtable,keys,action,reward,next_keys,terminal,alpha):
        old=self._nested_q(qtable,*keys,action=action); next_best=0.0 if terminal else max(self._nested_q(qtable,*next_keys,action=a) for a in ACTIONS)
        target=float(reward) if terminal else float(reward)+float(self.settings['discount'])*next_best; new=old+float(alpha)*(target-old)
        self._set_nested(qtable,keys,action,new); oldv=self._nested_v(vtable,*keys,action=action); self._set_nested(vtable,keys,action,oldv+1)
        return old,new

    def update(self,task_id,keys,action,reward,next_keys,terminal):
        if self.frozen:raise RuntimeError('Frozen CB19 policy cannot be updated')
        ag=float(self.settings.get('global_learning_rate',0.30)); ac=float(self.settings.get('context_learning_rate',0.25)); at=float(self.settings.get('task_learning_rate',0.15))
        g=self._update_component(self.q_global,self.visits_global,[keys['global']],action,reward,[next_keys['global']],terminal,ag)
        hc=self._update_component(self.q_hc,self.visits_hc,[keys['hc'],keys['global']],action,reward,[next_keys['hc'],next_keys['global']],terminal,ac)
        hd=self._update_component(self.q_hd,self.visits_hd,[keys['hd'],keys['global']],action,reward,[next_keys['hd'],next_keys['global']],terminal,ac)
        ot=self._update_component(self.q_ot,self.visits_ot,[keys['ot'],keys['global']],action,reward,[next_keys['ot'],next_keys['global']],terminal,ac)
        task=self._update_component(self.q_task,self.visits_task,[task_id,keys['task']],action,reward,[task_id,next_keys['task']],terminal,at)
        self.updates+=1
        return {'old_q_global':g[0],'new_q_global':g[1],'old_q_hc':hc[0],'new_q_hc':hc[1],'old_q_hd':hd[0],'new_q_hd':hd[1],'old_q_ot':ot[0],'new_q_ot':ot[1],'old_q_task':task[0],'new_q_task':task[1]}

    def to_dict(self):
        return {'namespace':'CB19','policy_schema_version':self.SCHEMA_VERSION,'seed':self.seed,'metadata':self.metadata,'q_global':self.q_global,'q_hc':self.q_hc,'q_hd':self.q_hd,'q_ot':self.q_ot,'q_task':self.q_task,'visits_global':self.visits_global,'visits_hc':self.visits_hc,'visits_hd':self.visits_hd,'visits_ot':self.visits_ot,'visits_task':self.visits_task,'decisions':self.decisions,'updates':self.updates,'frozen':self.frozen,'saved_at_utc':utc_now()}
    def save(self,path):write_json(path,self.to_dict())
    def freeze(self,destination):self.frozen=True; self.save(destination); return self.to_dict()
    def summary(self):return {'global_states':len(self.q_global),'hc_groups':len(self.q_hc),'hd_groups':len(self.q_hd),'ot_groups':len(self.q_ot),'task_memories':len(self.q_task),'decisions':self.decisions,'updates':self.updates,'frozen':self.frozen,'policy_schema_version':self.SCHEMA_VERSION}

QPolicy=HierarchicalQPolicy
