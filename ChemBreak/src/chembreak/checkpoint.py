from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .utils import utc_now


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS episodes(
              phase TEXT, stage INTEGER, assignment_id TEXT, status TEXT, terminal_ofs5 INTEGER,
              turns INTEGER, total_reward REAL, completion_reason TEXT, started_at TEXT, completed_at TEXT,
              PRIMARY KEY(phase,stage,assignment_id));
            CREATE TABLE IF NOT EXISTS turns(
              phase TEXT, stage INTEGER, assignment_id TEXT, context_id TEXT, turn_index INTEGER,
              context_turn_index INTEGER, target_query_index INTEGER, action_id TEXT,
              selection_mode TEXT, state_key TEXT, candidate_hash TEXT, candidate_source TEXT,
              candidate_gate_json TEXT, prompt TEXT, response TEXT, judgment_json TEXT,
              decision_json TEXT, transition_json TEXT, reward REAL, latency_seconds REAL,
              committed_at TEXT,
              PRIMARY KEY(phase,stage,assignment_id,turn_index));
            CREATE TABLE IF NOT EXISTS target_queries(
              query_index INTEGER PRIMARY KEY AUTOINCREMENT,
              phase TEXT, stage INTEGER, assignment_id TEXT, context_id TEXT, turn_index INTEGER,
              action_id TEXT, candidate_hash TEXT, prompt TEXT, response TEXT, latency_seconds REAL,
              judgment_status TEXT, judgment_json TEXT, created_at TEXT, judged_at TEXT);
            CREATE TABLE IF NOT EXISTS provider_events(
              event_index INTEGER PRIMARY KEY AUTOINCREMENT,
              phase TEXT, stage INTEGER, assignment_id TEXT, turn_index INTEGER,
              component TEXT, provider TEXT, event_type TEXT, error_category TEXT, metric TEXT,
              action_id TEXT, message TEXT, status_code TEXT, details_json TEXT, created_at TEXT);
            CREATE TABLE IF NOT EXISTS role_calls(
              call_index INTEGER PRIMARY KEY AUTOINCREMENT,
              phase TEXT, stage INTEGER, assignment_id TEXT, turn_index INTEGER,
              role TEXT, provider TEXT, model TEXT, status TEXT, metric TEXT,
              latency_seconds REAL, details_json TEXT, created_at TEXT);
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT);
            """
        )
        self.db.commit()

    @staticmethod
    def _enc(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def set_meta(self, key, value):
        encoded = self._enc(value)
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        if row and row["value"] == encoded:
            return
        self.db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (key, encoded))
        self.db.commit()

    def get_meta(self, key, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def start_episode(self, phase, stage, assignment_id):
        self.db.execute(
            "INSERT OR IGNORE INTO episodes VALUES(?,?,?,?,?,?,?,?,?,?)",
            (phase, int(stage), assignment_id, "running", 0, 0, 0.0, None, utc_now(), None),
        )
        self.db.commit()

    def episode_row(self, phase, stage, assignment_id):
        row = self.db.execute(
            "SELECT * FROM episodes WHERE phase=? AND stage=? AND assignment_id=?",
            (phase, int(stage), assignment_id),
        ).fetchone()
        return dict(row) if row else None

    def episode_done(self, phase, stage, assignment_id):
        row = self.episode_row(phase, stage, assignment_id)
        return bool(row and row["status"] == "complete")

    def reset_episode(self, phase, stage, assignment_id):
        with self.db:
            self.db.execute(
                "DELETE FROM turns WHERE phase=? AND stage=? AND assignment_id=?",
                (phase, int(stage), assignment_id),
            )
            self.db.execute(
                "DELETE FROM target_queries WHERE phase=? AND stage=? AND assignment_id=?",
                (phase, int(stage), assignment_id),
            )
            self.db.execute(
                "DELETE FROM provider_events WHERE phase=? AND stage=? AND assignment_id=?",
                (phase, int(stage), assignment_id),
            )
            self.db.execute(
                "DELETE FROM role_calls WHERE phase=? AND stage=? AND assignment_id=?",
                (phase, int(stage), assignment_id),
            )
            self.db.execute(
                "DELETE FROM episodes WHERE phase=? AND stage=? AND assignment_id=?",
                (phase, int(stage), assignment_id),
            )

    def rollback_uncommitted_query_artifacts(self, phase, stage, assignment_id):
        row = self.db.execute(
            "SELECT COALESCE(MAX(turn_index),0) AS last_turn FROM turns WHERE phase=? AND stage=? AND assignment_id=?",
            (phase, int(stage), assignment_id),
        ).fetchone()
        last_turn = int(row["last_turn"] if row else 0)
        with self.db:
            self.db.execute(
                """DELETE FROM target_queries
                   WHERE phase=? AND stage=? AND assignment_id=?
                   AND query_index NOT IN (
                     SELECT target_query_index FROM turns
                     WHERE phase=? AND stage=? AND assignment_id=?
                   )""",
                (phase, int(stage), assignment_id, phase, int(stage), assignment_id),
            )
            self.db.execute(
                "DELETE FROM provider_events WHERE phase=? AND stage=? AND assignment_id=? AND turn_index>?",
                (phase, int(stage), assignment_id, last_turn),
            )
            self.db.execute(
                "DELETE FROM role_calls WHERE phase=? AND stage=? AND assignment_id=? AND turn_index>?",
                (phase, int(stage), assignment_id, last_turn),
            )
        return last_turn

    def complete_episode(self, phase, stage, assignment_id, terminal_ofs5, turns, total_reward, reason):
        self.db.execute(
            """UPDATE episodes SET status=?,terminal_ofs5=?,turns=?,total_reward=?,completion_reason=?,completed_at=?
               WHERE phase=? AND stage=? AND assignment_id=?""",
            (
                "complete", int(bool(terminal_ofs5)), int(turns), float(total_reward), str(reason), utc_now(),
                phase, int(stage), assignment_id,
            ),
        )
        self.db.commit()

    def save_target_query(self, *, phase, stage, assignment_id, context_id, turn_index, action_id,
                          candidate_hash, prompt, response, latency_seconds):
        cursor = self.db.execute(
            """INSERT INTO target_queries(
                 phase,stage,assignment_id,context_id,turn_index,action_id,candidate_hash,prompt,response,
                 latency_seconds,judgment_status,judgment_json,created_at,judged_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                phase, int(stage), assignment_id, str(context_id), int(turn_index), str(action_id),
                str(candidate_hash), str(prompt), str(response), float(latency_seconds),
                "pending", "{}", utc_now(), None,
            ),
        )
        self.db.commit()
        return int(cursor.lastrowid)

    def finalize_target_query(self, query_index, judgment, status):
        encoded = self._enc(judgment or {})
        with self.db:
            self.db.execute(
                "UPDATE target_queries SET judgment_status=?,judgment_json=?,judged_at=? WHERE query_index=?",
                (str(status), encoded, utc_now(), int(query_index)),
            )
            self.db.execute(
                "UPDATE turns SET judgment_json=? WHERE target_query_index=?",
                (encoded, int(query_index)),
            )

    def _turn_values(self, row):
        return (
            row["phase"], int(row["stage"]), row["assignment_id"], str(row["context_id"]),
            int(row["turn_index"]), int(row["context_turn_index"]), int(row["target_query_index"]),
            str(row["action_id"]), str(row.get("selection_mode", "")), str(row.get("state_key", "")),
            str(row.get("candidate_hash", "")), str(row.get("candidate_source", "")),
            self._enc(row.get("candidate_gate", {})), str(row.get("prompt", "")), str(row.get("response", "")),
            self._enc(row.get("judgment", {})), self._enc(row.get("decision", {})),
            self._enc(row.get("transition", {})), float(row.get("reward", 0)),
            float(row.get("latency_seconds", 0)), utc_now(),
        )

    def save_turn(self, **row):
        values = self._turn_values(row)
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO turns VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                values,
            )

    def save_turn_and_checkpoint(self, row, *, active_checkpoint, policy_snapshot, memory_snapshot):
        values = self._turn_values(row)
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO turns VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                values,
            )
            self.db.execute(
                "INSERT OR REPLACE INTO meta VALUES(?,?)",
                ("active_stage_checkpoint", self._enc(active_checkpoint)),
            )
            self.db.execute(
                "INSERT OR REPLACE INTO meta VALUES(?,?)",
                ("task_policy_snapshot", self._enc(policy_snapshot)),
            )
            self.db.execute(
                "INSERT OR REPLACE INTO meta VALUES(?,?)",
                ("task_memory_snapshot", self._enc(memory_snapshot)),
            )

    def save_provider_event(self, phase, stage, assignment_id, *, turn_index, component, provider,
                            event_type, error_category="", metric="", action_id="", message="",
                            status_code="", details=None):
        cursor = self.db.execute(
            """INSERT INTO provider_events(
                 phase,stage,assignment_id,turn_index,component,provider,event_type,error_category,metric,
                 action_id,message,status_code,details_json,created_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                str(phase), int(stage), str(assignment_id), int(turn_index), str(component), str(provider),
                str(event_type), str(error_category), str(metric), str(action_id), str(message),
                str(status_code or ""), self._enc(details or {}), utc_now(),
            ),
        )
        self.db.commit()
        return int(cursor.lastrowid)

    def save_role_call(self, phase, stage, assignment_id, *, turn_index, role, provider, model,
                       status, metric="", latency_seconds=0.0, details=None):
        cursor = self.db.execute(
            """INSERT INTO role_calls(
                 phase,stage,assignment_id,turn_index,role,provider,model,status,metric,latency_seconds,
                 details_json,created_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                str(phase), int(stage), str(assignment_id), int(turn_index), str(role), str(provider),
                str(model), str(status), str(metric), float(latency_seconds), self._enc(details or {}), utc_now(),
            ),
        )
        self.db.commit()
        return int(cursor.lastrowid)

    def episodes(self):
        return [dict(row) for row in self.db.execute(
            "SELECT * FROM episodes ORDER BY assignment_id,phase,stage"
        ).fetchall()]

    def turns(self):
        return [dict(row) for row in self.db.execute(
            "SELECT * FROM turns ORDER BY assignment_id,phase,stage,turn_index"
        ).fetchall()]

    def target_queries(self):
        return [dict(row) for row in self.db.execute(
            "SELECT * FROM target_queries ORDER BY query_index"
        ).fetchall()]

    def provider_events(self):
        return [dict(row) for row in self.db.execute(
            "SELECT * FROM provider_events ORDER BY event_index"
        ).fetchall()]

    def role_calls(self):
        return [dict(row) for row in self.db.execute(
            "SELECT * FROM role_calls ORDER BY call_index"
        ).fetchall()]

    def contains_mock_markers(self):
        row = self.db.execute(
            "SELECT 1 FROM target_queries WHERE prompt LIKE '%[MOCK %' OR response LIKE '%[MOCK %' LIMIT 1"
        ).fetchone()
        return bool(row)

    def close(self):
        self.db.close()
