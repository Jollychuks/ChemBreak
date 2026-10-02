from __future__ import annotations
import json,math,random
from pathlib import Path
from .constants import ACTIONS
from .utils import stable_hex,utc_now,write_json


class QPolicy:
    """Compact task-local two-level Q policy carried across all three episodes."""
    SCHEMA_VERSION=4

    def __init__(self,settings,seed,data=None):
        self.settings=settings; self.seed=int(seed); data=data or {}
        if data and int(data.get('policy_schema_version',0))!=self.SCHEMA_VERSION:raise RuntimeError('CB29 policy schema mismatch; use fresh CB29 storage')
        self.q_task={a:float(v) for a,v in dict(data.get('q_task',{})).items()}
        self.q_state={str(k):{a:float(v) for a,v in dict(row).items()} for k,row in dict(data.get('q_state',{})).items()}
        self.visits_task={a:int(v) for a,v in dict(data.get('visits_task',{})).items()}
        self.visits_state={str(k):{a:int(v) for a,v in dict(row).items()} for k,row in dict(data.get('visits_state',{})).items()}
        self.nonpositive_streak={a:int(v) for a,v in dict(data.get('nonpositive_streak',{})).items()}
        self.decisions=int(data.get('decisions',0)); self.updates=int(data.get('updates',0)); self.assignment_id=str(data.get('assignment_id','')); self.metadata=dict(data.get('metadata',{}))

    def bind_task(self,task_id):
        task_id=str(task_id)
        if self.assignment_id and self.assignment_id!=task_id:raise RuntimeError(f'CB29 task-isolation violation: {self.assignment_id} -> {task_id}')
        self.assignment_id=task_id

    def _assert(self,task_id):
        if not self.assignment_id:self.assignment_id=str(task_id)
        if self.assignment_id!=str(task_id):raise RuntimeError('CB29 cross-task policy use is forbidden')

    def _qt(self,action):return float(self.q_task.get(action,0.0))
    def _qs(self,key,action):return float(self.q_state.get(str(key),{}).get(action,0.0))
    def _vt(self,action):return int(self.visits_task.get(action,0))
    def _vs(self,key,action):return int(self.visits_state.get(str(key),{}).get(action,0))

    def _effective_epsilon(self,key,base,recent):
        e=float(base)
        if sum(self._vs(key,a) for a in ACTIONS)==0:e+=float(self.settings.get('novel_state_epsilon_bonus',0))
        if recent and float(recent[-1].get('reward',0))<=0:e+=float(self.settings.get('negative_feedback_epsilon_bonus',0))
        return min(float(self.settings.get('max_effective_epsilon',.35)),max(0.0,e))

    def select(self,task_id,keys,epsilon,recent=None,extra_blocked=None):
        self._assert(task_id); key=str(keys['task']); recent=list(recent or []); blocked=set(extra_blocked or [])
        hard=max(1,int(self.settings.get('hard_block_after_nonpositive_repeats',2)))
        stagnant=[a for a in ACTIONS if a not in blocked and int(self.nonpositive_streak.get(a,0))>=hard]
        if len([a for a in ACTIONS if a not in blocked])>len(stagnant):blocked.update(stagnant)
        choices=[a for a in ACTIONS if a not in blocked]
        if not choices:return {'action':None,'mode':'no_available_action','base_epsilon':float(epsilon),'effective_epsilon':0.0,'blocked_actions':sorted(blocked),'task_key':key,'q_task_before':0.0,'q_state_before':0.0,'combined_q':0.0,'ucb_bonus':0.0,'repeat_penalty':0.0,'task_visits':0,'state_visits':0}
        self.decisions+=1; idx=self.decisions; eff=self._effective_epsilon(key,epsilon,recent)
        state_weight=float(self.settings.get('state_value_weight',.65)); task_weight=1.0-state_weight
        repeat_window=max(1,int(self.settings.get('repeat_window',3))); repeat_scale=float(self.settings.get('repeat_penalty',.30)); recent_actions=[str(x.get('action','')) for x in recent[-repeat_window:]]
        total_visits=sum(self._vt(a) for a in ACTIONS); ucb_scale=float(self.settings.get('ucb_exploration_scale',.20)); metrics={}
        for action in choices:
            repeat_penalty=repeat_scale*recent_actions.count(action); ucb=ucb_scale*math.sqrt(math.log(2+total_visits)/(1+self._vt(action)))
            combined=task_weight*self._qt(action)+state_weight*self._qs(key,action)+ucb-repeat_penalty
            metrics[action]={'q_task':self._qt(action),'q_state':self._qs(key,action),'combined':combined,'ucb':ucb,'repeat_penalty':repeat_penalty,'task_visits':self._vt(action),'state_visits':self._vs(key,action)}
        supported=[a for a in choices if self._vt(a)>0 or self._vs(key,a)>0]; rng=random.Random(self.seed+idx*7919)
        if rng.random()<eff:
            minimum=min(metrics[a]['task_visits'] for a in choices); pool=[a for a in choices if metrics[a]['task_visits']==minimum]; action=min(pool,key=lambda a:stable_hex(self.seed,idx,task_id,key,a)); mode='exploration'
        elif not supported:
            minimum=min(metrics[a]['task_visits'] for a in choices); pool=[a for a in choices if metrics[a]['task_visits']==minimum]; action=min(pool,key=lambda a:stable_hex(self.seed,task_id,key,a)); mode='cold_start'
        else:
            best=max(metrics[a]['combined'] for a in choices); pool=[a for a in choices if abs(metrics[a]['combined']-best)<1e-12]; action=min(pool,key=lambda a:(metrics[a]['task_visits'],stable_hex(self.seed,task_id,key,a))); mode='exploitation'
        m=metrics[action]
        return {'action':action,'mode':mode,'base_epsilon':float(epsilon),'effective_epsilon':eff,'blocked_actions':sorted(blocked),'task_key':key,'q_task_before':m['q_task'],'q_state_before':m['q_state'],'combined_q':m['combined'],'ucb_bonus':m['ucb'],'repeat_penalty':m['repeat_penalty'],'task_visits':m['task_visits'],'state_visits':m['state_visits']}

    def update(self,task_id,keys,action,reward,next_keys,done):
        self._assert(task_id); key=str(keys['task']); next_key=str(next_keys['task']); alpha=float(self.settings['learning_rate']); gamma=float(self.settings['discount'])
        old_task=self._qt(action); old_state=self._qs(key,action); next_best=0.0 if done else max(self._qs(next_key,a) for a in ACTIONS)
        state_target=float(reward) if done else float(reward)+gamma*next_best; new_state=old_state+alpha*(state_target-old_state); new_task=old_task+alpha*(float(reward)-old_task)
        self.q_task[action]=new_task; self.q_state.setdefault(key,{})[action]=new_state; self.visits_task[action]=self._vt(action)+1; self.visits_state.setdefault(key,{})[action]=self._vs(key,action)+1
        self.nonpositive_streak[action]=0 if float(reward)>0 else int(self.nonpositive_streak.get(action,0))+1; self.updates+=1
        return {'q_task_before':old_task,'q_task_after':new_task,'q_state_before':old_state,'q_state_after':new_state,'task_visits_after':self._vt(action),'state_visits_after':self._vs(key,action),'nonpositive_streak_after':self.nonpositive_streak[action],'learning_update_skipped':False}

    def to_dict(self):return {'namespace':'CB29','policy_schema_version':self.SCHEMA_VERSION,'seed':self.seed,'assignment_id':self.assignment_id,'metadata':self.metadata,'q_task':self.q_task,'q_state':self.q_state,'visits_task':self.visits_task,'visits_state':self.visits_state,'nonpositive_streak':self.nonpositive_streak,'decisions':self.decisions,'updates':self.updates,'saved_at_utc':utc_now()}
    def save(self,path):write_json(path,self.to_dict())
    @classmethod
    def load(cls,path,settings,seed):
        p=Path(path); return cls(settings,seed,json.loads(p.read_text()) if p.exists() else None)
