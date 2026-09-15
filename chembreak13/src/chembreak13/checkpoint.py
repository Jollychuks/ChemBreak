from __future__ import annotations
import json, sqlite3
from pathlib import Path
from typing import Any
from .utils import utc_now

class Store:
    def __init__(self,path):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True); self.db=sqlite3.connect(self.path); self.db.row_factory=sqlite3.Row
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS episodes(phase TEXT, epoch INTEGER, assignment_id TEXT, status TEXT, success INTEGER, turns INTEGER, total_reward REAL, terminal_reason TEXT, started_at TEXT, completed_at TEXT, PRIMARY KEY(phase,epoch,assignment_id));
        CREATE TABLE IF NOT EXISTS turns(phase TEXT, epoch INTEGER, assignment_id TEXT, turn_index INTEGER, action_id TEXT, selection_mode TEXT, state_key TEXT, q_general REAL, q_task REAL, combined_q REAL, prompt TEXT, response TEXT, judge_json TEXT, reward REAL, latency_seconds REAL, PRIMARY KEY(phase,epoch,assignment_id,turn_index));
        CREATE TABLE IF NOT EXISTS baseline_profiles(assignment_id TEXT PRIMARY KEY, response_class TEXT, goal_progress REAL, task_fidelity REAL, chemistry_relevance REAL, success INTEGER, judge_json TEXT, response TEXT);
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
        '''); self.db.commit()
    def set_meta(self,k,v): self.db.execute('INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)',(k,json.dumps(v))); self.db.commit()
    def get_meta(self,k,default=None):
        r=self.db.execute('SELECT value FROM meta WHERE key=?',(k,)).fetchone(); return json.loads(r['value']) if r else default
    def episode_done(self,phase,epoch,aid):
        r=self.db.execute('SELECT status FROM episodes WHERE phase=? AND epoch=? AND assignment_id=?',(phase,epoch,aid)).fetchone(); return bool(r and r['status']=='complete')
    def start_episode(self,phase,epoch,aid):
        self.db.execute('INSERT OR IGNORE INTO episodes VALUES(?,?,?,?,?,?,?,?,?,?)',(phase,epoch,aid,'running',0,0,0.0,None,utc_now(),None)); self.db.commit()
    def complete_episode(self,phase,epoch,aid,success,turns,total_reward,terminal):
        self.db.execute('UPDATE episodes SET status=?,success=?,turns=?,total_reward=?,terminal_reason=?,completed_at=? WHERE phase=? AND epoch=? AND assignment_id=?',('complete',int(success),turns,total_reward,terminal,utc_now(),phase,epoch,aid)); self.db.commit()
    def save_turn(self,**r):
        self.db.execute('''INSERT OR REPLACE INTO turns(phase,epoch,assignment_id,turn_index,action_id,selection_mode,state_key,q_general,q_task,combined_q,prompt,response,judge_json,reward,latency_seconds) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',(
            r['phase'],r['epoch'],r['assignment_id'],r['turn_index'],r['action_id'],r.get('selection_mode',''),r.get('state_key',''),r.get('q_general',0.0),r.get('q_task',0.0),r.get('combined_q',0.0),r['prompt'],r['response'],json.dumps(r['judge']),r['reward'],r.get('latency_seconds',0.0))); self.db.commit()
    def get_turns(self,phase,epoch,aid): return [dict(x) for x in self.db.execute('SELECT * FROM turns WHERE phase=? AND epoch=? AND assignment_id=? ORDER BY turn_index',(phase,epoch,aid)).fetchall()]
    def save_baseline(self,aid,response,judge):
        self.db.execute('INSERT OR REPLACE INTO baseline_profiles VALUES(?,?,?,?,?,?,?,?)',(aid,judge['response_class'],judge['goal_progress'],judge['task_fidelity'],judge['chemistry_relevance'],int(judge['success']),json.dumps(judge),response)); self.db.commit()
    def baseline(self,aid):
        r=self.db.execute('SELECT * FROM baseline_profiles WHERE assignment_id=?',(aid,)).fetchone(); return dict(r) if r else None
    def episodes(self): return [dict(x) for x in self.db.execute('SELECT * FROM episodes ORDER BY phase,epoch,assignment_id').fetchall()]
    def turns(self): return [dict(x) for x in self.db.execute('SELECT * FROM turns ORDER BY phase,epoch,assignment_id,turn_index').fetchall()]
    def close(self): self.db.close()
