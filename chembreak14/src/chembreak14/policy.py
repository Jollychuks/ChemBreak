from __future__ import annotations
import json, math, random
from pathlib import Path
from typing import Any
from .constants import ACTIONS
from .utils import stable_hex, utc_now, write_json

class QPolicy:
    def __init__(self, settings: dict[str,Any], seed: int, data: dict|None=None):
        self.settings=settings; self.seed=int(seed)
        data=data or {}
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
    def _qt(self,task,a): return float(self.q_task.get(task,{}).get(a,0.0))
    def combined(self,task,key,a): return self._qg(key,a)+float(self.settings.get('task_weight',1.0))*self._qt(task,a)
    def _visits(self, task,key,a): return int(self.visits_general.get(key,{}).get(a,0))+int(self.visits_task.get(task,{}).get(a,0))

    def select(self, task_id: str, key: str, epsilon: float, actions=None):
        actions=list(actions or ACTIONS); self.decisions+=1
        rng=random.Random(self.seed + self.decisions*7919)
        explore=(not self.frozen) and (rng.random() < float(epsilon))
        if explore:
            minv=min(self._visits(task_id,key,a) for a in actions)
            pool=[a for a in actions if self._visits(task_id,key,a)==minv]
            action=min(pool,key=lambda a: stable_hex(self.seed,self.decisions,task_id,key,a))
            mode='exploration'
        else:
            scores={a:self.combined(task_id,key,a) for a in actions}; best=max(scores.values())
            pool=[a for a,v in scores.items() if abs(v-best)<1e-12]
            action=min(pool,key=lambda a:(self._visits(task_id,key,a),stable_hex(self.seed,task_id,key,a)))
            mode='exploitation'
        return action, mode, self._qg(key,action), self._qt(task_id,action), self.combined(task_id,key,action)

    def update(self, task_id: str, key: str, action: str, reward: float, next_key: str, terminal: bool):
        if self.frozen: raise RuntimeError('Frozen CB14 policy cannot be updated')
        alpha=float(self.settings['learning_rate']); gamma=float(self.settings['discount'])
        next_g=max([self._qg(next_key,a) for a in ACTIONS],default=0.0)
        next_t=max([self._qt(task_id,a) for a in ACTIONS],default=0.0)
        oldg=self._qg(key,action); oldt=self._qt(task_id,action)
        targetg=float(reward) if terminal else float(reward)+gamma*next_g
        targett=float(reward) if terminal else float(reward)+gamma*next_t
        newg=oldg+alpha*(targetg-oldg); newt=oldt+alpha*(targett-oldt)
        self.q_general.setdefault(key,{})[action]=newg; self.q_task.setdefault(task_id,{})[action]=newt
        self.visits_general.setdefault(key,{})[action]=int(self.visits_general.get(key,{}).get(action,0))+1
        self.visits_task.setdefault(task_id,{})[action]=int(self.visits_task.get(task_id,{}).get(action,0))+1
        self.updates+=1
        return {'old_q_general':oldg,'new_q_general':newg,'old_q_task':oldt,'new_q_task':newt}

    def to_dict(self):
        return {'namespace':'CB14','seed':self.seed,'metadata':self.metadata,'q_general':self.q_general,'q_task':self.q_task,'visits_general':self.visits_general,'visits_task':self.visits_task,'decisions':self.decisions,'updates':self.updates,'frozen':self.frozen,'saved_at_utc':utc_now()}
    def save(self,path): write_json(path,self.to_dict())
    def freeze(self, destination): self.frozen=True; self.save(destination); return self.to_dict()
    def summary(self): return {'general_states':len(self.q_general),'task_memories':len(self.q_task),'decisions':self.decisions,'updates':self.updates,'frozen':self.frozen}
