from __future__ import annotations
from dataclasses import dataclass,field
from typing import Any
from .utils import progress_bin,fidelity_bin

def _turn_stage(i:int)->str:
    if i<=0:return 'start'
    if i==1:return 'early'
    return 'late'

def _trend(hist:list[dict[str,Any]])->str:
    if not hist:return 'start'
    d=float(hist[-1].get('progress_after',0))-float(hist[-1].get('progress_before',0))
    return 'improved' if d>=.10 else ('regressed' if d<=-.10 else 'flat')

def _reward_sign(value:float,hist:list[dict[str,Any]])->str:
    if not hist:return 'start'
    return 'positive' if float(value)>0 else 'nonpositive'

@dataclass
class EpisodeState:
    assignment_id:str; hc_id:str; hd_id:str; ot_id:str; max_turns:int
    turn_index:int=0; response_class:str='initial'; progress:float=0.0; fidelity:float=1.0
    previous_action:str='NONE'; previous_reward:float=0.0
    history:list[dict[str,str]]=field(default_factory=list)
    decision_history:list[dict[str,Any]]=field(default_factory=list)
    route_actions:list[str]=field(default_factory=list)
    peak_progress:float=0.0

    @classmethod
    def initial(cls,task,max_turns:int):
        return cls(str(task['assignment_id']),str(task['hc_id']),str(task['hd_id']),str(task['ot_id']),int(max_turns))
    def global_key(self)->str:
        return '|'.join([f'resp={self.response_class}',f'prog={progress_bin(self.progress)}',f'fid={fidelity_bin(self.fidelity)}',f'trend={_trend(self.decision_history)}',f'reward={_reward_sign(self.previous_reward,self.decision_history)}',f'stage={_turn_stage(self.turn_index)}'])
    def task_key(self)->str:return '|'.join([f'resp={self.response_class}',f'prog={progress_bin(self.progress)}',f'stage={_turn_stage(self.turn_index)}'])
    def policy_keys(self):return {'global':self.global_key(),'task':self.task_key(),'hc':self.hc_id,'hd':self.hd_id,'ot':self.ot_id}
    def advance(self,action,prompt,response,judge,reward):
        before=float(self.progress); self.turn_index+=1; self.previous_action=str(action); self.previous_reward=float(reward)
        self.response_class=str(judge['response_class']); self.progress=float(judge['goal_progress']); self.fidelity=float(judge['task_fidelity']); self.peak_progress=max(self.peak_progress,self.progress)
        self.history.extend([{'role':'user','content':prompt},{'role':'assistant','content':response}]); self.route_actions.append(str(action))
        self.decision_history.append({'action':str(action),'reward':float(reward),'success':bool(judge.get('final_success',False)),'progress_before':before,'progress_after':self.progress})
