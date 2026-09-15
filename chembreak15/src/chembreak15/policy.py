from __future__ import annotations
import json, random
from pathlib import Path
from typing import Any
from .constants import ACTIONS
from .utils import stable_hex, utc_now, write_json

class QPolicy:
    """CB15 tabular policy with state-aware task memory and anti-lock-in controls.

    CB14 used q_task[task][action], which allowed one historically successful
    action to dominate every state of a task. CB15 uses
    q_task[task][state][action] and combines it with the general state table.

    Selection also applies two conservative controls:
    * modest adaptive exploration in unseen/negative-feedback states;
    * a repetition penalty and a temporary block after repeated non-positive
      attempts with the same action.
    """
    SCHEMA_VERSION = 2

    def __init__(self, settings: dict[str,Any], seed: int, data: dict|None=None):
        self.settings=settings; self.seed=int(seed)
        data=data or {}
        schema=int(data.get('policy_schema_version', self.SCHEMA_VERSION if not data else 1))
        if data and schema != self.SCHEMA_VERSION:
            raise RuntimeError(
                f'CB15 policy schema mismatch: expected {self.SCHEMA_VERSION}, found {schema}. '
                'Do not reuse a CB14 policy artifact in CB15.'
            )
        self.q_general=data.get('q_general',{})
        self.q_task=data.get('q_task',{})
        self.visits_general=data.get('visits_general',{})
        self.visits_task=data.get('visits_task',{})
        self.decisions=int(data.get('decisions',0)); self.updates=int(data.get('updates',0)); self.frozen=bool(data.get('frozen',False))
        self.metadata=dict(data.get('metadata',{}))

    @classmethod
    def load(cls, path: str|Path, settings: dict[str,Any], seed: int):
        p=Path(path); data=json.loads(p.read_text()) if p.exists() else None; return cls(settings,seed,data)

    def _qg(self,key,a): return float(self.q_general.get(key,{}).get(a,0.0))
    def _qt(self,task,key,a): return float(self.q_task.get(task,{}).get(key,{}).get(a,0.0))
    def combined(self,task,key,a): return self._qg(key,a)+float(self.settings.get('task_weight',1.0))*self._qt(task,key,a)
    def _vg(self,key,a): return int(self.visits_general.get(key,{}).get(a,0))
    def _vt(self,task,key,a): return int(self.visits_task.get(task,{}).get(key,{}).get(a,0))
    def _visits(self,task,key,a): return self._vg(key,a)+self._vt(task,key,a)
    def _state_visits(self,task,key,actions): return sum(self._visits(task,key,a) for a in actions)

    @staticmethod
    def _trailing_nonpositive_repeats(recent: list[dict[str,Any]], action: str) -> int:
        count=0
        for item in reversed(recent):
            if str(item.get('action')) != action or float(item.get('reward',0.0)) > 0 or bool(item.get('success',False)):
                break
            count += 1
        return count

    def _effective_epsilon(self, task_id: str, key: str, base_epsilon: float, actions: list[str], recent: list[dict[str,Any]]):
        if self.frozen:
            return 0.0, {'novel_state_bonus':0.0,'negative_feedback_bonus':0.0}
        eps=float(base_epsilon)
        novelty=float(self.settings.get('novel_state_epsilon_bonus',0.0)) if self._state_visits(task_id,key,actions)==0 else 0.0
        negative=float(self.settings.get('negative_feedback_epsilon_bonus',0.0)) if recent and float(recent[-1].get('reward',0.0)) <= 0 else 0.0
        cap=float(self.settings.get('max_effective_epsilon',1.0))
        return min(cap,max(0.0,eps+novelty+negative)), {'novel_state_bonus':novelty,'negative_feedback_bonus':negative}

    def select(self, task_id: str, key: str, epsilon: float, actions=None, recent=None):
        actions=list(actions or ACTIONS); recent=list(recent or []); self.decisions+=1
        effective_epsilon, eps_parts=self._effective_epsilon(task_id,key,float(epsilon),actions,recent)
        hard_after=max(1,int(self.settings.get('hard_block_after_nonpositive_repeats',2)))
        blocked=[]
        if recent:
            last_action=str(recent[-1].get('action',''))
            if last_action in actions and self._trailing_nonpositive_repeats(recent,last_action) >= hard_after and len(actions)>1:
                blocked=[last_action]
        candidates=[a for a in actions if a not in blocked] or list(actions)

        repeat_unit=float(self.settings.get('repeat_nonpositive_penalty',0.0))
        raw_scores={a:self.combined(task_id,key,a) for a in actions}
        penalties={a:repeat_unit*self._trailing_nonpositive_repeats(recent,a) for a in actions}
        adjusted={a:raw_scores[a]-penalties[a] for a in actions}

        rng=random.Random(self.seed + self.decisions*7919)
        explore=(not self.frozen) and (rng.random() < effective_epsilon)
        if explore:
            minv=min(self._visits(task_id,key,a) for a in candidates)
            pool=[a for a in candidates if self._visits(task_id,key,a)==minv]
            action=min(pool,key=lambda a: stable_hex(self.seed,self.decisions,task_id,key,a))
            mode='exploration'
        else:
            best=max(adjusted[a] for a in candidates)
            pool=[a for a in candidates if abs(adjusted[a]-best)<1e-12]
            action=min(pool,key=lambda a:(self._visits(task_id,key,a),stable_hex(self.seed,task_id,key,a)))
            mode='exploitation'

        return {
            'action':action,'mode':mode,
            'base_epsilon':float(epsilon),'effective_epsilon':float(effective_epsilon),
            **eps_parts,'state_visits':int(self._state_visits(task_id,key,actions)),
            'q_general':self._qg(key,action),'q_task_state':self._qt(task_id,key,action),
            'combined_q':raw_scores[action],'repeat_penalty':penalties[action],
            'adjusted_score':adjusted[action],'blocked_actions':blocked,
        }

    def update(self, task_id: str, key: str, action: str, reward: float, next_key: str, terminal: bool):
        if self.frozen: raise RuntimeError('Frozen CB15 policy cannot be updated')
        alpha=float(self.settings['learning_rate']); gamma=float(self.settings['discount'])
        next_g=max([self._qg(next_key,a) for a in ACTIONS],default=0.0)
        next_t=max([self._qt(task_id,next_key,a) for a in ACTIONS],default=0.0)
        oldg=self._qg(key,action); oldt=self._qt(task_id,key,action)
        targetg=float(reward) if terminal else float(reward)+gamma*next_g
        targett=float(reward) if terminal else float(reward)+gamma*next_t
        newg=oldg+alpha*(targetg-oldg); newt=oldt+alpha*(targett-oldt)
        self.q_general.setdefault(key,{})[action]=newg
        self.q_task.setdefault(task_id,{}).setdefault(key,{})[action]=newt
        self.visits_general.setdefault(key,{})[action]=self._vg(key,action)+1
        self.visits_task.setdefault(task_id,{}).setdefault(key,{})[action]=self._vt(task_id,key,action)+1
        self.updates+=1
        return {'old_q_general':oldg,'new_q_general':newg,'old_q_task_state':oldt,'new_q_task_state':newt}

    def to_dict(self):
        return {'namespace':'CB15','policy_schema_version':self.SCHEMA_VERSION,'seed':self.seed,'metadata':self.metadata,'q_general':self.q_general,'q_task':self.q_task,'visits_general':self.visits_general,'visits_task':self.visits_task,'decisions':self.decisions,'updates':self.updates,'frozen':self.frozen,'saved_at_utc':utc_now()}
    def save(self,path): write_json(path,self.to_dict())
    def freeze(self, destination): self.frozen=True; self.save(destination); return self.to_dict()
    def summary(self):
        task_state_count=sum(len(states) for states in self.q_task.values())
        return {'general_states':len(self.q_general),'task_memories':len(self.q_task),'task_states':task_state_count,'decisions':self.decisions,'updates':self.updates,'frozen':self.frozen,'policy_schema_version':self.SCHEMA_VERSION}
