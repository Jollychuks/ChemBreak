from __future__ import annotations
from dataclasses import dataclass,field
from typing import Any


def _stage(i,max_turns):
    if i<=0:return 'start'
    if i==1:return 'early'
    if i>=max_turns-1:return 'late'
    return 'middle'


def _trend(hist):
    valid=[x for x in hist if x.get('judged')]
    if not valid:return 'start'
    d=int(valid[-1]['chcs_after'])-int(valid[-1]['chcs_before'])
    return 'up' if d>0 else ('down' if d<0 else 'flat')


@dataclass
class EpisodeState:
    assignment_id:str
    episode_index:int
    max_turns:int
    turn_index:int=0
    response_class:str='initial'
    chcs:int=1
    previous_action:str='NONE'
    previous_reward:float=0.0
    history:list[dict[str,str]]=field(default_factory=list)
    decision_history:list[dict[str,Any]]=field(default_factory=list)
    peak_chcs:int=1

    @classmethod
    def initial(cls,task,episode_index,max_turns,task_memory=None):
        latest=(task_memory or {}).get('latest_evaluated') or {}
        return cls(
            str(task['assignment_id']),int(episode_index),int(max_turns),
            response_class=str(latest.get('response_class','initial')),
            chcs=int(latest.get('chcs',1) or 1),
            previous_action=str(latest.get('action','NONE')),
            previous_reward=float(latest.get('reward',0.0) or 0.0),
            decision_history=list((task_memory or {}).get('decision_history',[])),
            peak_chcs=int((task_memory or {}).get('peak_chcs',latest.get('chcs',1)) or 1),
        )

    def chcs_trend(self):return _trend(self.decision_history)

    def task_key(self):
        return '|'.join([
            f'episode={self.episode_index}',f'resp={self.response_class}',f'chcs={self.chcs}',
            f'trend={self.chcs_trend()}',f'prev={self.previous_action}',f'stage={_stage(self.turn_index,self.max_turns)}',
        ])

    def policy_keys(self):return {'task':self.task_key()}

    def advance(self,action,prompt,response,judge,reward):
        before=self.chcs; before_class=self.response_class; self.turn_index+=1
        self.previous_action=str(action); self.previous_reward=float(reward)
        self.response_class=str(judge['response_class']); self.chcs=int(judge['chcs']); self.peak_chcs=max(self.peak_chcs,self.chcs)
        self.history.extend([{'role':'user','content':str(prompt)},{'role':'assistant','content':str(response)}])
        item={'episode':self.episode_index,'turn':self.turn_index,'action':str(action),'reward':float(reward),'success':bool(judge.get('final_success')),'judged':True,'chcs_before':before,'chcs_after':self.chcs,'response_class_before':before_class,'response_class_after':self.response_class}
        self.decision_history.append(item); return item

    def advance_unresolved(self,action,prompt,response,response_class='unknown'):
        before=self.chcs; before_class=self.response_class; self.turn_index+=1
        self.previous_action=str(action); self.previous_reward=0.0; self.response_class=str(response_class)
        self.history.extend([{'role':'user','content':str(prompt)},{'role':'assistant','content':str(response)}])
        item={'episode':self.episode_index,'turn':self.turn_index,'action':str(action),'reward':0.0,'success':False,'judged':False,'chcs_before':before,'chcs_after':before,'response_class_before':before_class,'response_class_after':self.response_class}
        self.decision_history.append(item); return item
