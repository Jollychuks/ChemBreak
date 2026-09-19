from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from .utils import progress_bin, fidelity_bin

def _turn_stage(i:int)->str:
    if i<=0:return 'start'
    if i==1:return 'early'
    if i==2:return 'middle'
    return 'late'

def _trend(hist:list[dict[str,Any]])->str:
    if not hist:return 'start'
    d=float(hist[-1].get('quality_after',0))-float(hist[-1].get('quality_before',0))
    return 'improved' if d>=.10 else ('regressed' if d<=-.10 else 'flat')

def _reward_sign(value:float,hist:list[dict[str,Any]])->str:
    if not hist:return 'start'
    return 'positive' if float(value)>0 else 'nonpositive'

@dataclass
class EpisodeState:
    assignment_id:str
    max_turns:int
    turn_index:int=0
    response_class:str='initial'
    progress:float=0.0
    fidelity:float=1.0
    correctness:float=1.0
    quality_progress:float=0.0
    previous_action:str='NONE'
    previous_reward:float=0.0
    history:list[dict[str,str]]=field(default_factory=list)
    decision_history:list[dict[str,Any]]=field(default_factory=list)
    route_actions:list[str]=field(default_factory=list)
    route_steps:list[dict[str,Any]]=field(default_factory=list)
    peak_progress:float=0.0
    peak_quality:float=0.0

    @classmethod
    def initial(cls,task,max_turns:int):
        return cls(str(task['assignment_id']),int(max_turns))
    def global_key(self)->str:
        return '|'.join([
            f'resp={self.response_class}',
            f'prog={progress_bin(self.progress)}',
            f'fid={fidelity_bin(self.fidelity)}',
            f'corr={fidelity_bin(self.correctness)}',
            f'trend={_trend(self.decision_history)}',
            f'reward={_reward_sign(self.previous_reward,self.decision_history)}',
            f'stage={_turn_stage(self.turn_index)}',
        ])
    def task_key(self)->str:
        return '|'.join([
            f'resp={self.response_class}',
            f'prog={progress_bin(self.progress)}',
            f'corr={fidelity_bin(self.correctness)}',
            f'stage={_turn_stage(self.turn_index)}',
        ])
    def policy_keys(self):
        return {'global':self.global_key(),'task':self.task_key()}
    def advance(self,action,prompt,response,judge,reward,candidate_gate=None,source='fresh'):
        before=float(self.quality_progress)
        self.turn_index+=1
        self.previous_action=str(action)
        self.previous_reward=float(reward)
        self.response_class=str(judge['response_class'])
        self.progress=float(judge['goal_progress'])
        self.fidelity=float(judge['task_fidelity'])
        self.correctness=float(judge['response_correctness'])
        self.quality_progress=float(judge.get('quality_progress',self.progress*self.fidelity*self.correctness))
        self.peak_progress=max(self.peak_progress,self.progress)
        self.peak_quality=max(self.peak_quality,self.quality_progress)
        self.history.extend([{'role':'user','content':prompt},{'role':'assistant','content':response}])
        self.route_actions.append(str(action))
        self.route_steps.append({
            'action':str(action),
            'prompt':str(prompt),
            'candidate_gate':dict(candidate_gate or {}),
            'source':str(source),
        })
        self.decision_history.append({
            'action':str(action),'reward':float(reward),'success':bool(judge.get('final_success',False)),
            'quality_before':before,'quality_after':self.quality_progress,
            'progress_after':self.progress,
        })
