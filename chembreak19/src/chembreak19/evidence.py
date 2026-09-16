from __future__ import annotations
import hashlib, json, math, re, unicodedata
from pathlib import Path
from typing import Any
from .utils import utc_now, write_json

class EvidenceMemory:
    """Persistent task-level evidence for exact realized candidates.

    Exact candidates are identified by a hash of normalized realized text.  The
    abstract MDP action remains a separate field so CB19 never conflates the
    action with one particular realization.
    """
    SCHEMA_VERSION = 1

    def __init__(self, settings: dict[str,Any], data: dict|None=None):
        self.settings=settings
        data=data or {}
        schema=int(data.get('evidence_schema_version', self.SCHEMA_VERSION if not data else 0))
        if data and schema != self.SCHEMA_VERSION:
            raise RuntimeError(f'CB19 evidence schema mismatch: expected {self.SCHEMA_VERSION}, found {schema}.')
        self.tasks=dict(data.get('tasks',{}))
        self.metadata=dict(data.get('metadata',{}))
        self.frozen=bool(data.get('frozen',False))
        self.records=int(data.get('records',0))

    @staticmethod
    def normalize_candidate(text: str) -> str:
        text=unicodedata.normalize('NFKC',str(text)).strip()
        return re.sub(r'\s+',' ',text)

    @classmethod
    def realization_id(cls,text: str) -> str:
        """Stable ID for the exact normalized attack-LLM utterance."""
        return hashlib.sha256(cls.normalize_candidate(text).encode('utf-8')).hexdigest()[:24]

    @classmethod
    def candidate_id(cls,text: str, action_id: str) -> str:
        """Action-conditioned candidate ID.

        The textual realization has its own ``realization_id``.  Including the
        abstract action here prevents an identical utterance produced under two
        different MDP actions from being silently conflated, while the separate
        realization ID still lets the per-episode mask avoid repeating the same
        exact text.
        """
        payload=f'{str(action_id)}\n{cls.normalize_candidate(text)}'
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]

    @classmethod
    def load(cls,path,settings):
        p=Path(path)
        return cls(settings,json.loads(p.read_text()) if p.exists() else None)

    def _entry(self,task_id,candidate_id):
        return self.tasks.setdefault(str(task_id),{}).setdefault(str(candidate_id),{})

    def record(self, *, task_id: str, prompt: str, action_id: str, global_key: str, task_key: str,
               phase: str, epoch: int, turn_index: int, success: bool, reward: float,
               combined_q: float, source: str) -> dict[str,Any]:
        if self.frozen:
            raise RuntimeError('Frozen CB19 evidence memory cannot be updated')
        rid=self.realization_id(prompt); cid=self.candidate_id(prompt,action_id); e=self._entry(task_id,cid)
        if not e:
            e.update({
                'candidate_id':cid,'realization_id':rid,'prompt':self.normalize_candidate(prompt),'action_id':str(action_id),
                'attempts':0,'successes':0,'failures':0,'reward_sum':0.0,'best_reward':None,
                'q_sum':0.0,'q_observations':0,'first_seen':{'phase':phase,'epoch':int(epoch),'turn':int(turn_index)},
                'last_seen':None,'last_success':None,'global_state_keys':[],'task_state_keys':[],
                'epochs_seen':[],'success_epochs':[],'failure_epochs':[],'sources':[],
            })
        if e.get('action_id') != str(action_id) or e.get('realization_id') != rid:
            raise RuntimeError('CB19 evidence identity collision detected')
        e['attempts']+=1
        if success:
            e['successes']+=1; e['last_success']={'phase':phase,'epoch':int(epoch),'turn':int(turn_index)}
            if int(epoch)>0 and int(epoch) not in e['success_epochs']: e['success_epochs'].append(int(epoch))
        else:
            e['failures']+=1
            if int(epoch)>0 and int(epoch) not in e['failure_epochs']: e['failure_epochs'].append(int(epoch))
        e['reward_sum']=float(e['reward_sum'])+float(reward)
        e['best_reward']=float(reward) if e['best_reward'] is None else max(float(e['best_reward']),float(reward))
        e['q_sum']=float(e['q_sum'])+float(combined_q); e['q_observations']=int(e['q_observations'])+1
        e['last_seen']={'phase':phase,'epoch':int(epoch),'turn':int(turn_index),'success':bool(success),'reward':float(reward)}
        for key,name in ((global_key,'global_state_keys'),(task_key,'task_state_keys')):
            if str(key) not in e[name]: e[name].append(str(key))
        if int(epoch)>0 and int(epoch) not in e['epochs_seen']: e['epochs_seen'].append(int(epoch))
        if str(source) not in e['sources']: e['sources'].append(str(source))
        for name in ('global_state_keys','task_state_keys','epochs_seen','success_epochs','failure_epochs','sources'):
            e[name]=sorted(e[name])
        self.records+=1
        return dict(e)

    def _wilson_lower(self,s:int,n:int) -> float:
        if n<=0:return 0.0
        z=float(self.settings.get('wilson_z',1.96)); p=float(s)/float(n); zz=z*z
        center=p+zz/(2*n); margin=z*math.sqrt((p*(1-p)+zz/(4*n))/n); denom=1+zz/n
        return max(0.0,(center-margin)/denom)

    @staticmethod
    def _bounded_quality(value: float, scale: float) -> float:
        scale=max(abs(float(scale)),1e-9)
        return 0.5+0.5*math.tanh(float(value)/scale)

    def score(self,e:dict[str,Any]) -> dict[str,float]:
        n=max(0,int(e.get('attempts',0))); s=max(0,int(e.get('successes',0)))
        reliability=self._wilson_lower(s,n)
        target=max(1,int(self.settings.get('target_support_attempts',3)))
        support=min(1.0, math.log1p(n)/math.log1p(target)) if n else 0.0
        mean_reward=float(e.get('reward_sum',0.0))/n if n else 0.0
        qn=max(0,int(e.get('q_observations',0))); mean_q=float(e.get('q_sum',0.0))/qn if qn else 0.0
        reward_quality=self._bounded_quality(mean_reward,float(self.settings.get('reward_scale',2.0)))
        q_quality=self._bounded_quality(mean_q,float(self.settings.get('q_scale',2.0)))
        weights={
            'reliability':float(self.settings.get('reliability_weight',0.55)),
            'support':float(self.settings.get('support_weight',0.20)),
            'reward':float(self.settings.get('reward_weight',0.15)),
            'q':float(self.settings.get('q_weight',0.10)),
        }
        denom=sum(weights.values()) or 1.0
        rank=(weights['reliability']*reliability+weights['support']*support+weights['reward']*reward_quality+weights['q']*q_quality)/denom
        return {'rank_score':float(rank),'wilson_lower':float(reliability),'support_score':float(support),'mean_reward':float(mean_reward),'mean_q':float(mean_q),'reward_quality':float(reward_quality),'q_quality':float(q_quality)}

    def rank_candidates(self,task_id: str, *, require_success: bool=True) -> list[dict[str,Any]]:
        rows=[]
        for cid,e in self.tasks.get(str(task_id),{}).items():
            if require_success and int(e.get('successes',0))<1: continue
            row={**e,**self.score(e)}; rows.append(row)
        rows.sort(key=lambda x:(-float(x['rank_score']),-int(x.get('successes',0)),-int(x.get('attempts',0)),str(x['candidate_id'])))
        for i,row in enumerate(rows,1): row['rank']=i
        return rows

    def best_live_candidate(self,task_id: str, keys: dict[str,str], *, attempted_realizations: set[str]|None=None, action_filter: set[str]|None=None):
        attempted_realizations=attempted_realizations or set(); candidates=self.rank_candidates(task_id,require_success=True)
        require_task_match=bool(self.settings.get('require_task_state_match_for_replay',True))
        best=None; best_score=None
        for row in candidates:
            if row.get('realization_id') in attempted_realizations: continue
            if action_filter is not None and row['action_id'] not in action_filter: continue
            task_match=keys.get('task') in set(row.get('task_state_keys',[]))
            if require_task_match and not task_match: continue
            score=float(row['rank_score'])
            if keys.get('global') in set(row.get('global_state_keys',[])): score+=float(self.settings.get('state_match_global_bonus',0.03))
            if task_match: score+=float(self.settings.get('state_match_task_bonus',0.02))
            key=(score,int(row.get('successes',0)),int(row.get('attempts',0)),-int(row.get('rank',999999)))
            if best is None or key>best_score: best=row; best_score=key
        if best is None:return None
        return {**best,'selection_score':float(best_score[0])}

    def build_rankings(self) -> dict[str,list[dict[str,Any]]]:
        return {task:self.rank_candidates(task,require_success=True) for task in sorted(self.tasks)}

    def choose_frozen_candidate(self, rankings: dict[str,list[dict[str,Any]]], task_id: str, keys: dict[str,str], attempted_realizations: set[str]|None=None):
        """Choose an immutable remembered candidate only in a compatible coarse task state.

        Exact successful utterances can depend on the conversational state in which
        they were discovered.  CB19 therefore never replays a late/contextual
        candidate into an incompatible fresh state.  The frozen rank itself remains
        immutable; state compatibility only controls eligibility and tie-breaking.
        """
        attempted_realizations=attempted_realizations or set()
        require_task_match=bool(self.settings.get('require_task_state_match_for_replay',True))
        rows=[r for r in rankings.get(str(task_id),[]) if r.get('realization_id') not in attempted_realizations]
        if require_task_match:
            rows=[r for r in rows if keys.get('task') in set(r.get('task_state_keys',[]))]
        if not rows:return None
        best=None; best_key=None
        for row in rows:
            score=float(row['rank_score'])
            if keys.get('global') in set(row.get('global_state_keys',[])):
                score+=float(self.settings.get('state_match_global_bonus',0.03))
            if keys.get('task') in set(row.get('task_state_keys',[])):
                score+=float(self.settings.get('state_match_task_bonus',0.02))
            key=(score,-int(row.get('rank',999999)))
            if best is None or key>best_key:
                best={**row,'selection_score':float(score)}; best_key=key
        return best

    def coverage(self) -> dict[str,Any]:
        task_count=len(self.tasks); covered=sum(1 for task in self.tasks if any(int(e.get('successes',0))>0 for e in self.tasks[task].values()))
        candidates=sum(len(v) for v in self.tasks.values()); successful_candidates=sum(1 for v in self.tasks.values() for e in v.values() if int(e.get('successes',0))>0)
        return {'tasks_with_evidence':task_count,'tasks_with_successful_evidence':covered,'successful_evidence_coverage':(covered/task_count if task_count else 0.0),'candidate_count':candidates,'successful_candidate_count':successful_candidates,'records':self.records}

    def to_dict(self):
        return {'namespace':'CB19','evidence_schema_version':self.SCHEMA_VERSION,'metadata':self.metadata,'frozen':self.frozen,'records':self.records,'tasks':self.tasks,'saved_at_utc':utc_now()}
    def save(self,path): write_json(path,self.to_dict())
