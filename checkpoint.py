from __future__ import annotations
import json,sqlite3
from pathlib import Path
from .utils import utc_now

class Store:
    def __init__(self,path):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True); self.db=sqlite3.connect(self.path); self.db.row_factory=sqlite3.Row
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS episodes(
          phase TEXT, epoch INTEGER, assignment_id TEXT, status TEXT, success INTEGER,
          turns INTEGER, total_reward REAL, terminal_reason TEXT, started_at TEXT, completed_at TEXT,
          PRIMARY KEY(phase,epoch,assignment_id));
        CREATE TABLE IF NOT EXISTS turns(
          phase TEXT, epoch INTEGER, assignment_id TEXT, context_id TEXT, turn_index INTEGER,
          target_query_index INTEGER,
          action_id TEXT, selection_mode TEXT, state_key TEXT, route_id TEXT, route_rank REAL,
          q_global REAL, q_task REAL, combined_q REAL, route_bonus REAL,
          candidate_hash TEXT, candidate_source TEXT, candidate_gate_json TEXT,
          prompt TEXT, response TEXT, judge_json TEXT, decision_json TEXT,
          reward REAL, latency_seconds REAL,
          PRIMARY KEY(phase,epoch,assignment_id,context_id,turn_index));
        CREATE TABLE IF NOT EXISTS target_queries(
          query_index INTEGER PRIMARY KEY AUTOINCREMENT,
          phase TEXT, epoch INTEGER, assignment_id TEXT, context_id TEXT, turn_index INTEGER,
          action_id TEXT, candidate_hash TEXT, prompt TEXT, response TEXT, latency_seconds REAL,
          judge_status TEXT, judge_json TEXT, created_at TEXT, judged_at TEXT);
        CREATE TABLE IF NOT EXISTS provider_events(
          phase TEXT, epoch INTEGER, assignment_id TEXT, event_index INTEGER,
          stage TEXT, provider TEXT, event_type TEXT, error_code TEXT, action_id TEXT,
          message TEXT, details_json TEXT, created_at TEXT,
          PRIMARY KEY(phase,epoch,assignment_id,event_index));
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT);
        '''); self.db.commit()
    @staticmethod
    def _enc(v):return json.dumps(v,sort_keys=True,separators=(',',':'))
    def set_meta(self,k,v):
        e=self._enc(v); r=self.db.execute('SELECT value FROM meta WHERE key=?',(k,)).fetchone()
        if r and r['value']==e:return
        self.db.execute('INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)',(k,e)); self.db.commit()
    def get_meta(self,k,default=None):
        r=self.db.execute('SELECT value FROM meta WHERE key=?',(k,)).fetchone(); return json.loads(r['value']) if r else default
    def episode_done(self,phase,epoch,aid):
        r=self.db.execute('SELECT status FROM episodes WHERE phase=? AND epoch=? AND assignment_id=?',(phase,int(epoch),aid)).fetchone(); return bool(r and r['status']=='complete')
    def episode_status(self,phase,epoch,aid):
        r=self.db.execute('SELECT status FROM episodes WHERE phase=? AND epoch=? AND assignment_id=?',(phase,int(epoch),aid)).fetchone(); return r['status'] if r else None
    def delete_episode_and_turns(self,phase,epoch,aid):
        # Raw target_queries are intentionally retained as immutable audit evidence of real calls.
        with self.db:
            self.db.execute('DELETE FROM turns WHERE phase=? AND epoch=? AND assignment_id=?',(phase,int(epoch),aid))
            self.db.execute('DELETE FROM provider_events WHERE phase=? AND epoch=? AND assignment_id=?',(phase,int(epoch),aid))
            self.db.execute('DELETE FROM episodes WHERE phase=? AND epoch=? AND assignment_id=?',(phase,int(epoch),aid))
    def start_episode(self,phase,epoch,aid):
        self.db.execute('INSERT OR IGNORE INTO episodes VALUES(?,?,?,?,?,?,?,?,?,?)',(phase,int(epoch),aid,'running',0,0,0.0,None,utc_now(),None)); self.db.commit()
    def complete_episode(self,phase,epoch,aid,success,turns,total_reward,terminal):
        self.db.execute('UPDATE episodes SET status=?,success=?,turns=?,total_reward=?,terminal_reason=?,completed_at=? WHERE phase=? AND epoch=? AND assignment_id=?',('complete',int(success),int(turns),float(total_reward),str(terminal),utc_now(),phase,int(epoch),aid)); self.db.commit()
    def complete_episode_with_learning(self,phase,epoch,aid,success,turns,total_reward,terminal,policy_snapshot,route_snapshot):
        with self.db:
            self.db.execute('UPDATE episodes SET status=?,success=?,turns=?,total_reward=?,terminal_reason=?,completed_at=? WHERE phase=? AND epoch=? AND assignment_id=?',('complete',int(success),int(turns),float(total_reward),str(terminal),utc_now(),phase,int(epoch),aid))
            self.db.execute('INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)',('training_policy_snapshot',self._enc(policy_snapshot)))
            self.db.execute('INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)',('training_route_snapshot',self._enc(route_snapshot)))
    def save_target_query(self,*,phase,epoch,assignment_id,context_id,turn_index,action_id,candidate_hash,prompt,response,latency_seconds):
        cur=self.db.execute('INSERT INTO target_queries(phase,epoch,assignment_id,context_id,turn_index,action_id,candidate_hash,prompt,response,latency_seconds,judge_status,judge_json,created_at,judged_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(phase,int(epoch),assignment_id,str(context_id),int(turn_index),str(action_id),str(candidate_hash),str(prompt),str(response),float(latency_seconds),'pending','{}',utc_now(),None)); self.db.commit(); return int(cur.lastrowid)
    def finalize_target_query(self,query_index,judge,status='judged'):
        self.db.execute('UPDATE target_queries SET judge_status=?,judge_json=?,judged_at=? WHERE query_index=?',(str(status),self._enc(judge or {}),utc_now(),int(query_index))); self.db.commit()
    def mark_target_query_judge_error(self,query_index,error):
        self.db.execute('UPDATE target_queries SET judge_status=?,judge_json=?,judged_at=? WHERE query_index=?',('judge_error',self._enc({'error':str(error)}),utc_now(),int(query_index))); self.db.commit()
    def _turn_vals(self,r):
        return (r['phase'],int(r['epoch']),r['assignment_id'],str(r.get('context_id','main')),int(r['turn_index']),int(r.get('target_query_index',0) or 0),r['action_id'],r.get('selection_mode',''),r.get('state_key',''),r.get('route_id',''),r.get('route_rank'),r.get('q_global',0.0),r.get('q_task',0.0),r.get('combined_q',0.0),r.get('route_bonus',0.0),r.get('candidate_hash',''),r.get('candidate_source',''),self._enc(r.get('candidate_gate',{})),r.get('prompt',''),r.get('response',''),self._enc(r.get('judge',{})),self._enc(r.get('decision',{})),float(r.get('reward',0)),float(r.get('latency_seconds',0)))
    def save_turn(self,**r):
        with self.db:self.db.execute('INSERT OR REPLACE INTO turns VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',self._turn_vals(r))
    def save_turn_with_learning(self,policy_snapshot,route_snapshot,**r):
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO turns VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',self._turn_vals(r))
            self.db.execute('INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)',('training_policy_snapshot',self._enc(policy_snapshot)))
            self.db.execute('INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)',('training_route_snapshot',self._enc(route_snapshot)))
    def save_provider_event(self,phase,epoch,aid,*,stage,provider,event_type,error_code,action_id='',message='',details=None):
        idx=1+len(self.get_provider_events(phase,epoch,aid)); self.db.execute('INSERT OR REPLACE INTO provider_events VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(phase,int(epoch),aid,idx,stage,provider,event_type,error_code,action_id,message,self._enc(details or {}),utc_now())); self.db.commit(); return idx
    def get_provider_events(self,phase,epoch,aid):return [dict(x) for x in self.db.execute('SELECT * FROM provider_events WHERE phase=? AND epoch=? AND assignment_id=? ORDER BY event_index',(phase,int(epoch),aid)).fetchall()]
    def provider_events(self):return [dict(x) for x in self.db.execute('SELECT * FROM provider_events ORDER BY phase,epoch,assignment_id,event_index').fetchall()]
    def episodes(self):return [dict(x) for x in self.db.execute('SELECT * FROM episodes ORDER BY phase,epoch,assignment_id').fetchall()]
    def turns(self):return [dict(x) for x in self.db.execute('SELECT * FROM turns ORDER BY phase,epoch,assignment_id,turn_index').fetchall()]
    def target_queries(self):return [dict(x) for x in self.db.execute('SELECT * FROM target_queries ORDER BY query_index').fetchall()]
    def clear_uncommitted_episodes(self):
        with self.db:self.db.execute("DELETE FROM episodes WHERE status!='complete'")
    def close(self):self.db.close()
