from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from .utils import progress_bin, fidelity_bin

@dataclass
class EpisodeState:
    assignment_id: str
    hc_id: str
    hd_id: str
    ot_id: str
    max_turns: int
    baseline_class: str
    baseline_progress: float
    baseline_fidelity: float
    turn_index: int = 0
    response_class: str = 'initial'
    progress: float = 0.0
    fidelity: float = 1.0
    previous_action: str = 'NONE'
    previous_reward: float = 0.0
    history: list[dict[str,str]] = field(default_factory=list)

    @classmethod
    def initial(cls, task: dict[str,Any], baseline: dict[str,Any], max_turns: int):
        return cls(
            assignment_id=str(task['assignment_id']), hc_id=str(task['hc_id']), hd_id=str(task['hd_id']), ot_id=str(task['ot_id']),
            max_turns=max_turns, baseline_class=str(baseline.get('response_class','unknown')),
            baseline_progress=float(baseline.get('goal_progress',0.0)), baseline_fidelity=float(baseline.get('task_fidelity',1.0)),
            response_class=str(baseline.get('response_class','initial')), progress=float(baseline.get('goal_progress',0.0)),
            fidelity=float(baseline.get('task_fidelity',1.0)),
        )

    def key(self) -> str:
        return '|'.join([
            self.hc_id,self.hd_id,self.ot_id,
            f"resp={self.response_class}",f"prog={progress_bin(self.progress)}",f"fid={fidelity_bin(self.fidelity)}",
            f"prev={self.previous_action}",f"turn={self.turn_index}",
        ])

    def advance(self, action: str, prompt: str, response: str, judge: dict[str,Any], reward: float):
        self.turn_index += 1; self.previous_action=action; self.previous_reward=float(reward)
        self.response_class=str(judge['response_class']); self.progress=float(judge['goal_progress']); self.fidelity=float(judge['task_fidelity'])
        self.history.extend([{'role':'user','content':prompt},{'role':'assistant','content':response}])
