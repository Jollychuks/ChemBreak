from __future__ import annotations
import json, sqlite3
from pathlib import Path
from typing import Any
from .utils import utc_now

class Store:
    def __init__(self,path):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.path); self.db.row_factory=sqlite3.Row
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS episodes(
          phase TEXT, epoch INTEGER, assignment_id TEXT, status TEXT, success INTEGER,
          turns INTEGER, total_reward REAL, terminal_reason TEXT, started_at TEXT, completed_at TEXT,
          PRIMARY KEY(phase,epoch,assignment_id));
        CREATE TABLE IF NOT EXISTS turns(
          phase TEXT, epoch INTEGER, assignment_id TEXT, turn_index INTEGER,
          action_id TEXT, selection_mode TEXT, state_key TEXT,
          q_global REAL, q_task REAL, combined_q REAL,
          candidate_id TEXT, realization_id TEXT, candidate_source TEXT, candidate_rank_score REAL,
          prompt TEXT, response TEXT, judge_json TEXT, decision_json TEXT,
          reward REAL, latency_seconds REAL,
          PRIMARY KEY(phase,epoch,assignment_id,turn_index));
        CREATE TABLE IF NOT EXISTS baseline_profiles(
          assignment_id TEXT PRIMARY KEY, response_class TEXT, goal_progress REAL,
          task_fidelity REAL, chemistry_relevance REAL, success INTEGER, judge_json TEXT, response TEXT);
        CREATE TABLE IF NOT EXISTS provider_events(
          phase TEXT, epoch INTEGER, assignment_id TEXT, event_index INTEGER,
          stage TEXT, provider TEXT, event_type TEXT, error_code TEXT, action_id TEXT,
          message TEXT, details_json TEXT, created_at TEXT,
          PRIMARY KEY(phase,epoch,assignment_id,event_index));
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
        '''); self.db.commit()

    @staticmethod
    def _encode(v):return json.dumps(v,sort_keys=True,separators=(',',':'))
    def set_meta(self,k,v):
        encoded=self._encode(v); r=self.db.execute('SELECT value FROM meta WHERE key=?',(k,)).fetchone()
        if r and r['value']==encoded:return
        self.db.execute('INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)',(k,encoded)); self.db.commit()
    def get_meta(self,k,default=None):
        r=self.db.execute('SELECT value FROM meta WHERE key=?',(k,)).fetchone(); return json.loads(r['value']) if r else default
    def episode_done(self,phase,epoch,aid):
        r=self.db.execute('SELECT status FROM episodes WHERE phase=? AND epoch=? AND assignment_id=?',(phase,epoch,aid)).fetchone(); return bool(r and r['status']=='complete')
    def start_episode(self,phase,epoch,aid):
        self.db.execute('INSERT OR IGNORE INTO episodes VALUES(?,?,?,?,?,?,?,?,?,?)',(phase,epoch,aid,'running',0,0,0.0,None,utc_now(),None)); self.db.commit()
    def complete_episode(self,phase,epoch,aid,success,turns,total_reward,terminal):
        self.db.execute('UPDATE episodes SET status=?,success=?,turns=?,total_reward=?,terminal_reason=?,completed_at=? WHERE phase=? AND epoch=? AND assignment_id=?',('complete',int(success),turns,total_reward,terminal,utc_now(),phase,epoch,aid)); self.db.commit()

    def _turn_values(self,r):
        return (r['phase'],r['epoch'],r['assignment_id'],r['turn_index'],r['action_id'],r.get('selection_mode',''),r.get('state_key',''),r.get('q_global',0.0),r.get('q_task',0.0),r.get('combined_q',0.0),r.get('candidate_id',''),r.get('realization_id',''),r.get('candidate_source',''),r.get('candidate_rank_score'),r['prompt'],r['response'],json.dumps(r['judge'],sort_keys=True),json.dumps(r.get('decision',{}),sort_keys=True),r['reward'],r.get('latency_seconds',0.0))
    def _insert_turn(self,r):
        self.db.execute('''INSERT OR REPLACE INTO turns(
          phase,epoch,assignment_id,turn_index,action_id,selection_mode,state_key,q_global,q_task,combined_q,
          candidate_id,realization_id,candidate_source,candidate_rank_score,prompt,response,judge_json,decision_json,reward,latency_seconds)
          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',self._turn_values(r))
    def save_turn(self,**r):
        with self.db:self._insert_turn(r)
    def save_turn_with_learning(self,policy_snapshot: dict[str,Any],evidence_snapshot: dict[str,Any],**r):
        with self.db:
            self._insert_turn(r)
            self.db.execute('INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)',('training_policy_snapshot',self._encode(policy_snapshot)))
            self.db.execute('INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)',('training_evidence_snapshot',self._encode(evidence_snapshot)))
    def get_turns(self,phase,epoch,aid):return [dict(x) for x in self.db.execute('SELECT * FROM turns WHERE phase=? AND epoch=? AND assignment_id=? ORDER BY turn_index',(phase,epoch,aid)).fetchall()]
    def save_baseline(self,aid,response,judge):
        self.db.execute('INSERT OR REPLACE INTO baseline_profiles VALUES(?,?,?,?,?,?,?,?)',(aid,judge['response_class'],judge['goal_progress'],judge['task_fidelity'],judge['chemistry_relevance'],int(judge['success']),json.dumps(judge,sort_keys=True),response)); self.db.commit()
    def baseline(self,aid):
        r=self.db.execute('SELECT * FROM baseline_profiles WHERE assignment_id=?',(aid,)).fetchone(); return dict(r) if r else None
    def provider_events(self):return [dict(x) for x in self.db.execute('SELECT * FROM provider_events ORDER BY phase,epoch,assignment_id,event_index').fetchall()]
    def get_provider_events(self,phase,epoch,aid):return [dict(x) for x in self.db.execute('SELECT * FROM provider_events WHERE phase=? AND epoch=? AND assignment_id=? ORDER BY event_index',(phase,epoch,aid)).fetchall()]
    def save_provider_event(self,phase,epoch,aid,*,stage,provider,event_type,error_code,action_id,message='',details=None):
        idx=1+len(self.get_provider_events(phase,epoch,aid))
        self.db.execute('INSERT OR REPLACE INTO provider_events VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(phase,epoch,aid,idx,stage,provider,event_type,error_code,action_id,message,self._encode(details or {}),utc_now())); self.db.commit(); return idx
    def episodes(self):return [dict(x) for x in self.db.execute('SELECT * FROM episodes ORDER BY phase,epoch,assignment_id').fetchall()]
    def turns(self):return [dict(x) for x in self.db.execute('SELECT * FROM turns ORDER BY phase,epoch,assignment_id,turn_index').fetchall()]
    def clear_uncommitted_episodes(self):
        with self.db:self.db.execute("DELETE FROM episodes WHERE status!='complete'")
    def close(self):self.db.close()
