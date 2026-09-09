from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .schema import TurnRecord
from .utils import utc_now


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS assets (
    assignment_id TEXT NOT NULL,
    condition TEXT NOT NULL,
    asset_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (assignment_id, condition)
);
CREATE TABLE IF NOT EXISTS episodes (
    episode_id TEXT PRIMARY KEY,
    assignment_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    condition TEXT NOT NULL,
    status TEXT NOT NULL,
    success INTEGER NOT NULL DEFAULT 0,
    queries_used INTEGER NOT NULL DEFAULT 0,
    cumulative_reward REAL NOT NULL DEFAULT 0,
    terminal_reason TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT
);
CREATE TABLE IF NOT EXISTS transcripts (
    episode_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    record_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (episode_id, turn_index)
);
CREATE TABLE IF NOT EXISTS evaluations (
    episode_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    record_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (episode_id, turn_index)
);
CREATE TABLE IF NOT EXISTS api_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id TEXT,
    record_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id TEXT,
    stage TEXT NOT NULL,
    error_type TEXT NOT NULL,
    error_message TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class CheckpointStore:
    def __init__(self, path: str | Path, signature: str, metadata: dict[str, Any]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)
        existing = self.get_metadata("run_signature")
        if existing and existing != signature:
            raise RuntimeError(
                "Checkpoint signature mismatch. Use a fresh output directory for the changed configuration."
            )
        if not existing:
            self.set_metadata("run_signature", signature)
            for key, value in metadata.items():
                self.set_metadata(key, value)

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()

    def set_metadata(self, key: str, value: Any) -> None:
        serialized = json.dumps(value, sort_keys=True, ensure_ascii=False)
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)", (key, serialized)
        )
        self.connection.commit()

    def get_metadata(self, key: str) -> Any | None:
        row = self.connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else None

    def get_asset(self, assignment_id: str, condition: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT asset_json FROM assets WHERE assignment_id=? AND condition=?",
            (assignment_id, condition),
        ).fetchone()
        return json.loads(row["asset_json"]) if row else None

    def put_asset(self, assignment_id: str, condition: str, asset: dict[str, Any]) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO assets VALUES (?, ?, ?, ?)",
            (assignment_id, condition, json.dumps(asset, ensure_ascii=False), utc_now()),
        )
        self.connection.commit()

    def episode_status(self, episode_id: str) -> str | None:
        row = self.connection.execute(
            "SELECT status FROM episodes WHERE episode_id=?", (episode_id,)
        ).fetchone()
        return str(row["status"]) if row else None

    def episode_queries_used(self, episode_id: str) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) AS n FROM transcripts WHERE episode_id=?", (episode_id,)
        ).fetchone()
        return int(row["n"] or 0)

    def start_episode(self, episode_id: str, assignment_id: str, target_id: str, condition: str) -> None:
        self.connection.execute(
            """INSERT OR IGNORE INTO episodes(
                episode_id, assignment_id, target_id, condition, status, started_at
            ) VALUES (?, ?, ?, ?, 'running', ?)""",
            (episode_id, assignment_id, target_id, condition, utc_now()),
        )
        self.connection.execute(
            """UPDATE episodes
               SET status='running', terminal_reason=NULL, completed_at=NULL
               WHERE episode_id=? AND status!='complete'""",
            (episode_id,),
        )
        self.connection.commit()

    def existing_turns(self, episode_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """SELECT t.record_json AS transcript_json, e.record_json AS evaluation_json
               FROM transcripts t
               JOIN evaluations e USING (episode_id, turn_index)
               WHERE t.episode_id=? ORDER BY t.turn_index""",
            (episode_id,),
        ).fetchall()
        return [
            {**json.loads(row["transcript_json"]), **json.loads(row["evaluation_json"])}
            for row in rows
        ]

    def pending_transcripts(self, episode_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """SELECT t.record_json
               FROM transcripts t
               LEFT JOIN evaluations e USING (episode_id, turn_index)
               WHERE t.episode_id=? AND e.episode_id IS NULL
               ORDER BY t.turn_index""",
            (episode_id,),
        ).fetchall()
        return [json.loads(row["record_json"]) for row in rows]

    def save_transcript(self, turn: TurnRecord) -> None:
        """Durably save a consumed target query before any judge is called."""
        transcript = {
            "episode_id": turn.episode_id,
            "assignment_id": turn.assignment_id,
            "target_id": turn.target_id,
            "condition": turn.condition,
            "turn_index": turn.turn_index,
            "action_id": turn.action_id,
            "action_reason": turn.action_reason,
            "attack_prompt": turn.attack_prompt,
            "target_response": turn.target_response,
            "query_attempted": turn.query_attempted,
            "query_error": turn.query_error,
            "latency_seconds": turn.latency_seconds,
        }
        self.connection.execute(
            "INSERT OR IGNORE INTO transcripts VALUES (?, ?, ?, ?)",
            (
                turn.episode_id,
                turn.turn_index,
                json.dumps(transcript, ensure_ascii=False),
                utc_now(),
            ),
        )
        self.connection.execute(
            """UPDATE episodes
               SET queries_used=(SELECT COUNT(*) FROM transcripts WHERE episode_id=?)
               WHERE episode_id=?""",
            (turn.episode_id, turn.episode_id),
        )
        self.connection.commit()

    def save_evaluation(self, turn: TurnRecord) -> None:
        evaluation = {
            "episode_id": turn.episode_id,
            "turn_index": turn.turn_index,
            "safety_result": turn.safety_result,
            "chemistry_result": turn.chemistry_result,
            "adjudication_result": turn.adjudication_result,
            "deterministic_result": turn.deterministic_result,
            "reward": turn.reward,
            "success": turn.success,
            "terminal_reason": turn.terminal_reason,
        }
        transcript = self.connection.execute(
            "SELECT 1 FROM transcripts WHERE episode_id=? AND turn_index=?",
            (turn.episode_id, turn.turn_index),
        ).fetchone()
        if transcript is None:
            raise RuntimeError("Cannot save an evaluation before its target transcript.")
        self.connection.execute(
            "INSERT OR REPLACE INTO evaluations VALUES (?, ?, ?, ?)",
            (
                turn.episode_id,
                turn.turn_index,
                json.dumps(evaluation, ensure_ascii=False),
                utc_now(),
            ),
        )
        self.connection.execute(
            """UPDATE episodes
               SET cumulative_reward=(SELECT COALESCE(SUM(json_extract(record_json, '$.reward')),0)
                                      FROM evaluations WHERE episode_id=?),
                   queries_used=(SELECT COUNT(*) FROM transcripts WHERE episode_id=?)
               WHERE episode_id=?""",
            (turn.episode_id, turn.episode_id, turn.episode_id),
        )
        self.connection.commit()

    def save_turn(self, turn: TurnRecord) -> None:
        self.save_transcript(turn)
        self.save_evaluation(turn)

    def save_api_calls(self, episode_id: str | None, calls: list[dict[str, Any]]) -> None:
        for call in calls:
            self.connection.execute(
                "INSERT INTO api_calls(episode_id, record_json, created_at) VALUES (?, ?, ?)",
                (episode_id, json.dumps(call, ensure_ascii=False), utc_now()),
            )
        self.connection.commit()

    def finish_episode(self, episode_id: str, success: bool, terminal_reason: str) -> None:
        self.connection.execute(
            """UPDATE episodes SET status='complete', success=?, terminal_reason=?, completed_at=?
               WHERE episode_id=?""",
            (int(success), terminal_reason, utc_now(), episode_id),
        )
        self.connection.commit()

    def fail_episode(self, episode_id: str, error: Exception) -> None:
        self.connection.execute(
            """UPDATE episodes SET status='failed', terminal_reason=?, completed_at=?
               WHERE episode_id=?""",
            (f"{type(error).__name__}: {str(error)[:1000]}", utc_now(), episode_id),
        )
        self.connection.commit()

    def mark_pending_judgment(self, episode_id: str, error: Exception) -> None:
        self.connection.execute(
            """UPDATE episodes SET status='pending_judgment', terminal_reason=?, completed_at=NULL
               WHERE episode_id=?""",
            (f"{type(error).__name__}: {str(error)[:1000]}", episode_id),
        )
        self.connection.commit()

    def mark_ready(self, episode_id: str) -> None:
        self.connection.execute(
            """UPDATE episodes SET status='ready', terminal_reason=NULL, completed_at=NULL
               WHERE episode_id=?""",
            (episode_id,),
        )
        self.connection.commit()

    def mark_target_unavailable(
        self,
        target_id: str,
        episodes: list[tuple[str, str, str]],
        error: Exception,
    ) -> None:
        reason = f"{type(error).__name__}: {str(error)[:1000]}"
        now = utc_now()
        for episode_id, assignment_id, condition in episodes:
            self.connection.execute(
                """INSERT INTO episodes(
                       episode_id, assignment_id, target_id, condition, status,
                       terminal_reason, started_at, completed_at
                   ) VALUES (?, ?, ?, ?, 'target_unavailable', ?, ?, ?)
                   ON CONFLICT(episode_id) DO UPDATE SET
                       status=CASE WHEN episodes.status='complete' THEN 'complete'
                                   ELSE 'target_unavailable' END,
                       terminal_reason=CASE WHEN episodes.status='complete' THEN episodes.terminal_reason
                                            ELSE excluded.terminal_reason END,
                       completed_at=CASE WHEN episodes.status='complete' THEN episodes.completed_at
                                         ELSE excluded.completed_at END""",
                (episode_id, assignment_id, target_id, condition, reason, now, now),
            )
        self.connection.commit()

    def record_failure(self, episode_id: str | None, stage: str, error: Exception) -> None:
        self.connection.execute(
            """INSERT INTO failures(episode_id, stage, error_type, error_message, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (episode_id, stage, type(error).__name__, str(error)[:4000], utc_now()),
        )
        self.connection.commit()

    def completed_episode_ids(self, condition: str | None = None) -> set[str]:
        if condition is None:
            query = "SELECT episode_id FROM episodes WHERE status='complete'"
            parameters: tuple[Any, ...] = ()
        else:
            query = "SELECT episode_id FROM episodes WHERE status='complete' AND condition=?"
            parameters = (condition,)
        rows = self.connection.execute(query, parameters).fetchall()
        return {str(row["episode_id"]) for row in rows}

    def pending_episode_rows(self, condition: str | None = None) -> list[dict[str, Any]]:
        where = "WHERE e.status='pending_judgment'"
        parameters: tuple[Any, ...] = ()
        if condition is not None:
            where += " AND e.condition=?"
            parameters = (condition,)
        rows = self.connection.execute(
            f"""SELECT DISTINCT e.episode_id, e.assignment_id, e.target_id, e.condition
                FROM episodes e
                JOIN transcripts t ON t.episode_id=e.episode_id
                LEFT JOIN evaluations v USING (episode_id, turn_index)
                {where} AND v.episode_id IS NULL
                ORDER BY e.condition, e.target_id, e.assignment_id""",
            parameters,
        ).fetchall()
        return [dict(row) for row in rows]

    def progress_snapshot(self, total: int, condition: str | None = None) -> dict[str, Any]:
        where = ""
        parameters: tuple[Any, ...] = ()
        if condition is not None:
            where = " WHERE condition=?"
            parameters = (condition,)
        summary = self.connection.execute(
            f"""SELECT
                   SUM(CASE WHEN status='complete' THEN 1 ELSE 0 END) AS completed,
                   SUM(CASE WHEN status='complete' THEN success ELSE 0 END) AS successes,
                   SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS technical_failed,
                   SUM(CASE WHEN status='target_unavailable' THEN 1 ELSE 0 END) AS target_unavailable,
                   SUM(CASE WHEN status='pending_judgment' THEN 1 ELSE 0 END) AS pending_judgment,
                   SUM(CASE WHEN status='ready' THEN 1 ELSE 0 END) AS ready,
                   COALESCE(SUM(queries_used), 0) AS queries
               FROM episodes{where}""",
            parameters,
        ).fetchone()
        if condition is None:
            failure_count = self.connection.execute(
                "SELECT COUNT(*) AS n FROM failures"
            ).fetchone()["n"]
            refusal_parameters: tuple[Any, ...] = ()
            refusal_condition = ""
        else:
            failure_count = self.connection.execute(
                """SELECT COUNT(*) AS n FROM failures f
                   JOIN episodes e ON e.episode_id=f.episode_id WHERE e.condition=?""",
                (condition,),
            ).fetchone()["n"]
            refusal_parameters = (condition,)
            refusal_condition = " AND p.condition=?"
        refusal_count = self.connection.execute(
            f"""SELECT COUNT(*) AS n FROM evaluations v
               JOIN episodes p ON p.episode_id=v.episode_id
               WHERE json_extract(v.record_json, '$.safety_result.response_class')
                     IN ('refusal', 'safe_redirect'){refusal_condition}""",
            refusal_parameters,
        ).fetchone()["n"]
        groups = self.connection.execute(
            f"""SELECT target_id, condition, COUNT(*) AS completed, SUM(success) AS successes,
                      AVG(queries_used) AS mean_queries
               FROM episodes WHERE status='complete'
               {('AND condition=?' if condition is not None else '')}
               GROUP BY target_id, condition ORDER BY target_id, condition""",
            parameters,
        ).fetchall()
        completed = int(summary["completed"] or 0)
        successes = int(summary["successes"] or 0)
        technical_failed = int(summary["technical_failed"] or 0)
        target_unavailable = int(summary["target_unavailable"] or 0)
        pending_judgment = int(summary["pending_judgment"] or 0)
        ready = int(summary["ready"] or 0)
        unjudged = int(
            self.connection.execute(
                f"""SELECT COUNT(*) AS n FROM transcripts t
                   JOIN episodes p ON p.episode_id=t.episode_id
                   LEFT JOIN evaluations e USING (episode_id, turn_index)
                   WHERE e.episode_id IS NULL
                   {('AND p.condition=?' if condition is not None else '')}""",
                parameters,
            ).fetchone()["n"]
            or 0
        )
        by_group = []
        for row in groups:
            row_completed = int(row["completed"] or 0)
            row_successes = int(row["successes"] or 0)
            by_group.append(
                {
                    "target_id": str(row["target_id"]),
                    "condition": str(row["condition"]),
                    "completed": row_completed,
                    "successes": row_successes,
                    "attack_success_rate": row_successes / row_completed if row_completed else 0.0,
                    "mean_queries": float(row["mean_queries"] or 0.0),
                }
            )
        return {
            "completed": completed,
            "evaluated": completed,
            "total_episodes": int(total),
            "successes": successes,
            "attack_success_rate": successes / completed if completed else 0.0,
            "evaluated_coverage": completed / int(total) if total else 0.0,
            "technical_failed": technical_failed,
            "target_unavailable": target_unavailable,
            "pending_judgment": pending_judgment,
            "ready": ready,
            "unjudged_target_responses": unjudged,
            "queries": int(summary["queries"] or 0),
            "refusals": int(refusal_count or 0),
            "failures": int(failure_count or 0),
            "by_target_condition": by_group,
        }

    def backup(self, destination: str | Path) -> Path:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.connection.commit()
        with sqlite3.connect(destination) as backup_connection:
            self.connection.backup(backup_connection)
        return destination


def sync_checkpoint_to_gcs(local_path: Path, gcs_uri: str | None) -> str | None:
    if not gcs_uri:
        return None
    if not gcs_uri.startswith("gs://"):
        raise ValueError("gcs_checkpoint_uri must start with gs://")
    from google.cloud import storage

    without_scheme = gcs_uri[5:]
    bucket_name, _, prefix = without_scheme.partition("/")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    object_name = f"{prefix.rstrip('/')}/{local_path.name}" if prefix else local_path.name
    bucket.blob(object_name).upload_from_filename(local_path)
    return f"gs://{bucket_name}/{object_name}"
