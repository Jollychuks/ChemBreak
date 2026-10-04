from __future__ import annotations
import json,sqlite3
from pathlib import Path
from .utils import utc_now


class Store:
    def __init__(self,path):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True); self.db=sqlite3.connect(self.path); self.db.row_factory=sqlite3.Row
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS episodes(
          phase TEXT, episode INTEGER, assignment_id TEXT, status TEXT, success INTEGER,
          turns INTEGER, total_reward REAL, completion_reason TEXT, started_at TEXT, completed_at TEXT,
          PRIMARY KEY(phase,episode,assignment_id));
        CREATE TABLE IF NOT EXISTS turns(
          phase TEXT, episode INTEGER, assignment_id TEXT, context_id TEXT, turn_index INTEGER,
          target_query_index INTEGER, action_id TEXT, selection_mode TEXT, state_key TEXT,
          q_task REAL, candidate_hash TEXT, candidate_source TEXT, candidate_gate_json TEXT,
          prompt TEXT, response TEXT, judge_json TEXT, decision_json TEXT,
          reward REAL, latency_seconds REAL,
          PRIMARY KEY(phase,episode,assignment_id,context_id,turn_index));
        CREATE TABLE IF NOT EXISTS target_queries(
          query_index INTEGER PRIMARY KEY AUTOINCREMENT,
          phase TEXT, episode INTEGER, assignment_id TEXT, context_id TEXT, turn_index INTEGER,
          action_id TEXT, candidate_hash TEXT, prompt TEXT, response TEXT, latency_seconds REAL,
          judge_status TEXT, judge_json TEXT, created_at TEXT, judged_at TEXT);
        CREATE TABLE IF NOT EXISTS provider_events(
          phase TEXT, episode INTEGER, assignment_id TEXT, event_index INTEGER,
          stage TEXT, provider TEXT, event_type TEXT, error_code TEXT, action_id TEXT,
          message TEXT, details_json TEXT, created_at TEXT,
          PRIMARY KEY(phase,episode,assignment_id,event_index));
        CREATE TABLE IF NOT EXISTS role_calls(
          call_index INTEGER PRIMARY KEY AUTOINCREMENT,
          phase TEXT, episode INTEGER, assignment_id TEXT, turn_index INTEGER,
          role TEXT, provider TEXT, model TEXT, status TEXT,
          latency_seconds REAL, candidate_slot INTEGER, details_json TEXT, created_at TEXT);
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT);
        '''); self.db.commit()

    @staticmethod
    def _enc(v):return json.dumps(v,sort_keys=True,separators=(',',':'))
    def set_meta(self,k,v):
        e=self._enc(v); row=self.db.execute('SELECT value FROM meta WHERE key=?',(k,)).fetchone()
        if row and row['value']==e:return
        self.db.execute('INSERT OR REPLACE INTO meta VALUES(?,?)',(k,e)); self.db.commit()
    def get_meta(self,k,default=None):
        row=self.db.execute('SELECT value FROM meta WHERE key=?',(k,)).fetchone(); return json.loads(row['value']) if row else default
    def episode_done(self,phase,episode,aid):
        row=self.db.execute('SELECT status FROM episodes WHERE phase=? AND episode=? AND assignment_id=?',(phase,int(episode),aid)).fetchone(); return bool(row and row['status']=='complete')
    def episode_row(self,phase,episode,aid):
        row=self.db.execute('SELECT * FROM episodes WHERE phase=? AND episode=? AND assignment_id=?',(phase,int(episode),aid)).fetchone(); return dict(row) if row else None
    def episode_status(self,phase,episode,aid):
        row=self.episode_row(phase,episode,aid); return row['status'] if row else None
    def start_episode(self,phase,episode,aid):
        self.db.execute('INSERT OR IGNORE INTO episodes VALUES(?,?,?,?,?,?,?,?,?,?)',(phase,int(episode),aid,'running',0,0,0.0,None,utc_now(),None)); self.db.commit()
    def delete_episode_and_turns(self,phase,episode,aid):
        with self.db:
            self.db.execute('DELETE FROM turns WHERE phase=? AND episode=? AND assignment_id=?',(phase,int(episode),aid))
            self.db.execute('DELETE FROM provider_events WHERE phase=? AND episode=? AND assignment_id=?',(phase,int(episode),aid))
            self.db.execute('DELETE FROM episodes WHERE phase=? AND episode=? AND assignment_id=?',(phase,int(episode),aid))
    def complete_episode(self,phase,episode,aid,success,turns,total_reward,reason):
        self.db.execute('UPDATE episodes SET status=?,success=?,turns=?,total_reward=?,completion_reason=?,completed_at=? WHERE phase=? AND episode=? AND assignment_id=?',('complete',int(success),int(turns),float(total_reward),str(reason),utc_now(),phase,int(episode),aid)); self.db.commit()
    def complete_episode_with_learning(self,phase,episode,aid,success,turns,total_reward,reason,policy_snapshot,memory_snapshot):
        with self.db:
            self.db.execute('UPDATE episodes SET status=?,success=?,turns=?,total_reward=?,completion_reason=?,completed_at=? WHERE phase=? AND episode=? AND assignment_id=?',('complete',int(success),int(turns),float(total_reward),str(reason),utc_now(),phase,int(episode),aid))
            self.db.execute('INSERT OR REPLACE INTO meta VALUES(?,?)',('task_policy_snapshot',self._enc(policy_snapshot)))
            self.db.execute('INSERT OR REPLACE INTO meta VALUES(?,?)',('task_memory_snapshot',self._enc(memory_snapshot)))
    def mark_episode_recovered_success(self,phase,episode,aid):
        self.db.execute("UPDATE episodes SET success=1,completion_reason='recovered_validated_success' WHERE phase=? AND episode=? AND assignment_id=?",(phase,int(episode),aid)); self.db.commit()
    def save_target_query(self,*,phase,episode,assignment_id,context_id,turn_index,action_id,candidate_hash,prompt,response,latency_seconds):
        cur=self.db.execute('INSERT INTO target_queries(phase,episode,assignment_id,context_id,turn_index,action_id,candidate_hash,prompt,response,latency_seconds,judge_status,judge_json,created_at,judged_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(phase,int(episode),assignment_id,str(context_id),int(turn_index),str(action_id),str(candidate_hash),str(prompt),str(response),float(latency_seconds),'pending','{}',utc_now(),None)); self.db.commit(); return int(cur.lastrowid)
    def finalize_target_query(self,query_index,judge,status='judged'):
        encoded=self._enc(judge or {})
        with self.db:
            self.db.execute('UPDATE target_queries SET judge_status=?,judge_json=?,judged_at=? WHERE query_index=?',(str(status),encoded,utc_now(),int(query_index)))
            self.db.execute('UPDATE turns SET judge_json=? WHERE target_query_index=?',(encoded,int(query_index)))
    def mark_target_query_judge_error(self,query_index,error,audit=None):
        payload={'error':str(error),'judge_provenance':dict(audit or {})}; self.finalize_target_query(query_index,payload,'judge_error')
    def save_turn(self,**r):
        vals=(r['phase'],int(r['episode']),r['assignment_id'],str(r.get('context_id','main')),int(r['turn_index']),int(r.get('target_query_index',0) or 0),str(r['action_id']),str(r.get('selection_mode','')),str(r.get('state_key','')),float(r.get('q_task',0)),str(r.get('candidate_hash','')),str(r.get('candidate_source','')),self._enc(r.get('candidate_gate',{})),str(r.get('prompt','')),str(r.get('response','')),self._enc(r.get('judge',{})),self._enc(r.get('decision',{})),float(r.get('reward',0)),float(r.get('latency_seconds',0)))
        with self.db:self.db.execute('INSERT OR REPLACE INTO turns VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',vals)
    def save_turn_with_learning(self,policy_snapshot,memory_snapshot,**r):
        with self.db:
            vals=(r['phase'],int(r['episode']),r['assignment_id'],str(r.get('context_id','main')),int(r['turn_index']),int(r.get('target_query_index',0) or 0),str(r['action_id']),str(r.get('selection_mode','')),str(r.get('state_key','')),float(r.get('q_task',0)),str(r.get('candidate_hash','')),str(r.get('candidate_source','')),self._enc(r.get('candidate_gate',{})),str(r.get('prompt','')),str(r.get('response','')),self._enc(r.get('judge',{})),self._enc(r.get('decision',{})),float(r.get('reward',0)),float(r.get('latency_seconds',0)))
            self.db.execute('INSERT OR REPLACE INTO turns VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',vals)
            self.db.execute('INSERT OR REPLACE INTO meta VALUES(?,?)',('task_policy_snapshot',self._enc(policy_snapshot)))
            self.db.execute('INSERT OR REPLACE INTO meta VALUES(?,?)',('task_memory_snapshot',self._enc(memory_snapshot)))
    def save_provider_event(self,phase,episode,aid,*,stage,provider,event_type,error_code,action_id='',message='',details=None):
        idx=1+len(self.get_provider_events(phase,episode,aid)); self.db.execute('INSERT OR REPLACE INTO provider_events VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(phase,int(episode),aid,idx,str(stage),str(provider),str(event_type),str(error_code),str(action_id),str(message),self._enc(details or {}),utc_now())); self.db.commit(); return idx
    def get_provider_events(self,phase,episode,aid):return [dict(x) for x in self.db.execute('SELECT * FROM provider_events WHERE phase=? AND episode=? AND assignment_id=? ORDER BY event_index',(phase,int(episode),aid)).fetchall()]
    def episodes(self):return [dict(x) for x in self.db.execute('SELECT * FROM episodes ORDER BY assignment_id,phase,episode').fetchall()]
    def turns(self):return [dict(x) for x in self.db.execute('SELECT * FROM turns ORDER BY assignment_id,phase,episode,turn_index').fetchall()]
    def target_queries(self):return [dict(x) for x in self.db.execute('SELECT * FROM target_queries ORDER BY query_index').fetchall()]
    def unresolved_target_queries(self):return [dict(x) for x in self.db.execute("SELECT * FROM target_queries WHERE judge_status!='judged' ORDER BY query_index").fetchall()]
    def contains_mock_markers(self):
        row=self.db.execute("SELECT 1 FROM target_queries WHERE prompt LIKE '%[MOCK %' OR response LIKE '%[MOCK %' LIMIT 1").fetchone()
        return bool(row)
    def provider_events(self):return [dict(x) for x in self.db.execute('SELECT * FROM provider_events ORDER BY assignment_id,phase,episode,event_index').fetchall()]
    def save_role_call(self,phase,episode,aid,*,turn_index,role,provider,model,status,latency_seconds,candidate_slot=0,details=None):
        cur=self.db.execute('INSERT INTO role_calls(phase,episode,assignment_id,turn_index,role,provider,model,status,latency_seconds,candidate_slot,details_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(str(phase),int(episode),str(aid),int(turn_index),str(role),str(provider),str(model),str(status),float(latency_seconds),int(candidate_slot),self._enc(details or {}),utc_now())); self.db.commit(); return int(cur.lastrowid)
    def role_calls(self):return [dict(x) for x in self.db.execute('SELECT * FROM role_calls ORDER BY call_index').fetchall()]
    def committed_target_query_ids(self):
        return {int(x['target_query_index']) for x in self.db.execute('''SELECT t.target_query_index FROM turns t JOIN episodes e ON e.phase=t.phase AND e.episode=t.episode AND e.assignment_id=t.assignment_id WHERE t.target_query_index>0 AND e.status='complete' ''').fetchall()}
    def clear_uncommitted_episodes(self):
        for row in self.db.execute("SELECT phase,episode,assignment_id FROM episodes WHERE status!='complete'").fetchall():self.delete_episode_and_turns(row['phase'],row['episode'],row['assignment_id'])
    def close(self):self.db.close()
