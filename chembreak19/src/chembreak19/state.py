from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from .utils import progress_bin, fidelity_bin

def _turn_stage(turn_index: int) -> str:
    if turn_index <= 0: return 'start'
    if turn_index == 1: return 'early'
    return 'late'

def _trend(history: list[dict[str,Any]]) -> str:
    if not history: return 'start'
    last=history[-1]; delta=float(last.get('progress_after',0.0))-float(last.get('progress_before',0.0))
    if delta >= 0.10: return 'improved'
    if delta <= -0.10: return 'regressed'
    return 'flat'

def _reward_sign(value: float, history: list[dict[str,Any]]) -> str:
    if not history: return 'start'
    return 'positive' if float(value)>0 else 'nonpositive'

@dataclass
class EpisodeState:
    assignment_id: str
    hc_id: str
    hd_id: str
    ot_id: str
    max_turns: int
    turn_index: int = 0
    response_class: str = 'initial'
    progress: float = 0.0
    fidelity: float = 1.0
    previous_action: str = 'NONE'
    previous_reward: float = 0.0
    history: list[dict[str,str]] = field(default_factory=list)
    decision_history: list[dict[str,Any]] = field(default_factory=list)

    @classmethod
    def initial(cls, task: dict[str,Any], max_turns: int):
        # Every Baseline, learning epoch, and Optimized evaluation is a fresh
        # conversational episode.  What persists across learning epochs is the
        # learned Q/evidence state, never the previous target conversation.
        return cls(
            assignment_id=str(task['assignment_id']),hc_id=str(task['hc_id']),hd_id=str(task['hd_id']),ot_id=str(task['ot_id']),max_turns=max_turns,
            response_class='initial',progress=0.0,fidelity=1.0,
        )

    def global_key(self) -> str:
        return '|'.join([
            f"resp={self.response_class}",f"prog={progress_bin(self.progress)}",f"fid={fidelity_bin(self.fidelity)}",
            f"trend={_trend(self.decision_history)}",f"reward={_reward_sign(self.previous_reward,self.decision_history)}",f"stage={_turn_stage(self.turn_index)}",
        ])

    def task_key(self) -> str:
        # Deliberately coarser than the global key so the same task can revisit it across epochs.
        return '|'.join([
            f"resp={self.response_class}",f"prog={progress_bin(self.progress)}",f"stage={_turn_stage(self.turn_index)}",
        ])

    def policy_keys(self) -> dict[str,str]:
        return {'global':self.global_key(),'task':self.task_key(),'hc':self.hc_id,'hd':self.hd_id,'ot':self.ot_id}

    def key(self) -> str:
        return self.global_key()

    def advance(self, action: str, prompt: str, response: str, judge: dict[str,Any], reward: float):
        previous_progress=float(self.progress); self.turn_index+=1; self.previous_action=action; self.previous_reward=float(reward)
        self.response_class=str(judge['response_class']); self.progress=float(judge['goal_progress']); self.fidelity=float(judge['task_fidelity'])
        self.history.extend([{'role':'user','content':prompt},{'role':'assistant','content':response}])
        self.decision_history.append({'action':str(action),'reward':float(reward),'success':bool(judge.get('success',False)),'progress_before':previous_progress,'progress_after':float(self.progress)})
