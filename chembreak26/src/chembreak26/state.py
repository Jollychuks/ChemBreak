from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


def _turn_stage(i:int,max_turns:int)->str:
    if i<=0:return 'start'
    if i==1:return 'early'
    if i>=max_turns-1:return 'late'
    return 'middle'


def _chcs_trend(hist:list[dict[str,Any]])->str:
    if not hist:return 'start'
    d=int(hist[-1].get('chcs_after',1))-int(hist[-1].get('chcs_before',1))
    return 'up' if d>0 else ('down' if d<0 else 'flat')


@dataclass
class EpisodeState:
    assignment_id:str
    max_turns:int
    turn_index:int=0
    response_class:str='initial'
    chcs:int=1
    previous_action:str='NONE'
    previous_reward:float=0.0
    history:list[dict[str,str]]=field(default_factory=list)
    decision_history:list[dict[str,Any]]=field(default_factory=list)
    route_actions:list[str]=field(default_factory=list)
    route_steps:list[dict[str,Any]]=field(default_factory=list)
    peak_chcs:int=1

    @classmethod
    def initial(cls,task,max_turns:int):
        return cls(str(task['assignment_id']),int(max_turns))

    def chcs_trend(self)->str:
        return _chcs_trend(self.decision_history)

    def global_key(self)->str:
        return '|'.join([
            f'resp={self.response_class}',
            f'chcs={int(self.chcs)}',
            f'trend={self.chcs_trend()}',
            f'prev={self.previous_action}',
            f'stage={_turn_stage(self.turn_index,self.max_turns)}',
        ])

    def task_key(self)->str:
        return '|'.join([
            f'chcs={int(self.chcs)}',
            f'trend={self.chcs_trend()}',
            f'prev={self.previous_action}',
            f'stage={_turn_stage(self.turn_index,self.max_turns)}',
        ])

    def policy_keys(self):
        return {'global':self.global_key(),'task':self.task_key()}

    def advance(self,action,prompt,response,judge,reward,candidate_gate=None,source='fresh'):
        before=int(self.chcs)
        self.turn_index+=1
        self.previous_action=str(action)
        self.previous_reward=float(reward)
        self.response_class=str(judge['response_class'])
        self.chcs=int(judge['chcs'])
        self.peak_chcs=max(int(self.peak_chcs),int(self.chcs))
        self.history.extend([{'role':'user','content':prompt},{'role':'assistant','content':response}])
        self.route_actions.append(str(action))
        self.route_steps.append({
            'action':str(action),
            'prompt':str(prompt),
            'candidate_gate':dict(candidate_gate or {}),
            'source':str(source),
        })
        self.decision_history.append({
            'action':str(action),
            'reward':float(reward),
            'success':bool(judge.get('final_success',False)),
            'chcs_before':before,
            'chcs_after':int(self.chcs),
        })
