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
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
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
    episode_id TEXT NOT NULL, turn_index INTEGER NOT NULL,
    record_json TEXT NOT NULL, created_at TEXT NOT NULL,
    PRIMARY KEY (episode_id, turn_index)
);
CREATE TABLE IF NOT EXISTS observations (
    episode_id TEXT NOT NULL, turn_index INTEGER NOT NULL,
    record_json TEXT NOT NULL, created_at TEXT NOT NULL,
    PRIMARY KEY (episode_id, turn_index)
);
CREATE TABLE IF NOT EXISTS evaluations (
    episode_id TEXT NOT NULL, turn_index INTEGER NOT NULL,
    record_json TEXT NOT NULL, created_at TEXT NOT NULL,
    PRIMARY KEY (episode_id, turn_index)
);
CREATE TABLE IF NOT EXISTS api_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id TEXT,
    record_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id TEXT,
    stage TEXT NOT NULL, error_type TEXT NOT NULL,
    error_message TEXT NOT NULL, created_at TEXT NOT NULL
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
            raise RuntimeError("Checkpoint signature mismatch. Use a fresh run directory.")
        if not existing:
            self.set_metadata("run_signature", signature)
            for key, value in metadata.items():
                self.set_metadata(key, value)

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()

    def set_metadata(self, key: str, value: Any) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)",
            (key, json.dumps(value, ensure_ascii=False, sort_keys=True)),
        )
        self.connection.commit()

    def get_metadata(self, key: str) -> Any | None:
        row = self.connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else None

    def start_episode(self, episode_id: str, assignment_id: str, target_id: str) -> None:
        self.connection.execute(
            """INSERT OR IGNORE INTO episodes(
                episode_id,assignment_id,target_id,condition,status,started_at
            ) VALUES (?,?,?,'C3_ADAPTIVE_MDP','running',?)""",
            (episode_id, assignment_id, target_id, utc_now()),
        )
        self.connection.execute(
            """UPDATE episodes SET status='running', terminal_reason=NULL, completed_at=NULL
               WHERE episode_id=? AND status!='complete'""", (episode_id,),
        )
        self.connection.commit()

    def episode_status(self, episode_id: str) -> str | None:
        row = self.connection.execute("SELECT status FROM episodes WHERE episode_id=?", (episode_id,)).fetchone()
        return str(row["status"]) if row else None

    def save_transcript(self, turn: TurnRecord) -> None:
        self.connection.execute(
            "INSERT OR IGNORE INTO transcripts VALUES (?,?,?,?)",
            (turn.episode_id, turn.turn_index, json.dumps(turn.to_dict(), ensure_ascii=False), utc_now()),
        )
        self.connection.execute(
            """UPDATE episodes SET queries_used=(
                SELECT COUNT(*) FROM transcripts WHERE episode_id=?
            ) WHERE episode_id=?""", (turn.episode_id, turn.episode_id),
        )
        self.connection.commit()

    def save_observation(self, episode_id: str, turn_index: int, record: dict[str, Any]) -> None:
        if self.get_transcript(episode_id, turn_index) is None:
            raise RuntimeError("Cannot save an observation before its transcript.")
        record = {"episode_id": episode_id, "turn_index": turn_index, **record}
        self.connection.execute(
            "INSERT OR REPLACE INTO observations VALUES (?,?,?,?)",
            (episode_id, turn_index, json.dumps(record, ensure_ascii=False), utc_now()),
        )
        self.connection.commit()

    def save_evaluation(self, episode_id: str, turn_index: int, record: dict[str, Any]) -> None:
        if self.get_observation(episode_id, turn_index) is None:
            raise RuntimeError("Cannot save an evaluation before its observation.")
        record = {"episode_id": episode_id, "turn_index": turn_index, **record}
        self.connection.execute(
            "INSERT OR REPLACE INTO evaluations VALUES (?,?,?,?)",
            (episode_id, turn_index, json.dumps(record, ensure_ascii=False), utc_now()),
        )
        self.connection.execute(
            """UPDATE episodes SET cumulative_reward=(
                SELECT COALESCE(SUM(json_extract(record_json,'$.reward')),0)
                FROM evaluations WHERE episode_id=?
            ) WHERE episode_id=?""", (episode_id, episode_id),
        )
        self.connection.commit()

    def _get_record(self, table: str, episode_id: str, turn_index: int) -> dict[str, Any] | None:
        row = self.connection.execute(
            f"SELECT record_json FROM {table} WHERE episode_id=? AND turn_index=?",
            (episode_id, turn_index),
        ).fetchone()
        return json.loads(row["record_json"]) if row else None

    def get_transcript(self, episode_id: str, turn_index: int) -> dict[str, Any] | None:
        return self._get_record("transcripts", episode_id, turn_index)

    def get_observation(self, episode_id: str, turn_index: int) -> dict[str, Any] | None:
        return self._get_record("observations", episode_id, turn_index)

    def get_evaluation(self, episode_id: str, turn_index: int) -> dict[str, Any] | None:
        return self._get_record("evaluations", episode_id, turn_index)

    def turn_rows(self, episode_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """SELECT t.record_json transcript_json, o.record_json observation_json,
                      e.record_json evaluation_json
               FROM transcripts t
               LEFT JOIN observations o USING(episode_id,turn_index)
               LEFT JOIN evaluations e USING(episode_id,turn_index)
               WHERE t.episode_id=? ORDER BY t.turn_index""", (episode_id,),
        ).fetchall()
        values = []
        for row in rows:
            value = json.loads(row["transcript_json"])
            value["observation"] = json.loads(row["observation_json"]) if row["observation_json"] else None
            value["evaluation"] = json.loads(row["evaluation_json"]) if row["evaluation_json"] else None
            values.append(value)
        return values

    def pending_stage(self, episode_id: str) -> tuple[str, dict[str, Any]] | None:
        for row in self.turn_rows(episode_id):
            if row["observation"] is None:
                return "observation", row
            if row["evaluation"] is None:
                return "verification", row
        return None

    def pending_episodes(self, target_id: str | None = None) -> list[dict[str, Any]]:
        where = "WHERE status IN ('running','pending_verification','ready')"
        params: tuple[Any, ...] = ()
        if target_id:
            where += " AND target_id=?"
            params = (target_id,)
        rows = self.connection.execute(f"SELECT * FROM episodes {where} ORDER BY episode_id", params).fetchall()
        return [dict(row) for row in rows]

    def mark_pending(self, episode_id: str, stage: str) -> None:
        self.connection.execute(
            "UPDATE episodes SET status='pending_verification', terminal_reason=? WHERE episode_id=?",
            (stage, episode_id),
        )
        self.connection.commit()

    def mark_ready(self, episode_id: str) -> None:
        self.connection.execute(
            "UPDATE episodes SET status='ready', terminal_reason=NULL, completed_at=NULL WHERE episode_id=?",
            (episode_id,),
        )
        self.connection.commit()

    def finish_episode(self, episode_id: str, success: bool, terminal_reason: str) -> None:
        self.connection.execute(
            """UPDATE episodes SET status='complete', success=?, terminal_reason=?, completed_at=?
               WHERE episode_id=?""", (int(success), terminal_reason, utc_now(), episode_id),
        )
        self.connection.commit()

    def record_failure(self, episode_id: str | None, stage: str, error: Exception) -> None:
        self.connection.execute(
            "INSERT INTO failures(episode_id,stage,error_type,error_message,created_at) VALUES (?,?,?,?,?)",
            (episode_id, stage, type(error).__name__, str(error)[:4000], utc_now()),
        )
        self.connection.commit()

    def save_api_calls(self, episode_id: str | None, calls: list[dict[str, Any]]) -> None:
        for call in calls:
            self.connection.execute(
                "INSERT INTO api_calls(episode_id,record_json,created_at) VALUES (?,?,?)",
                (episode_id, json.dumps(call, ensure_ascii=False), utc_now()),
            )
        self.connection.commit()

    def completed_episode_ids(self, target_id: str | None = None) -> set[str]:
        if target_id:
            rows = self.connection.execute(
                "SELECT episode_id FROM episodes WHERE status='complete' AND target_id=?", (target_id,)
            ).fetchall()
        else:
            rows = self.connection.execute("SELECT episode_id FROM episodes WHERE status='complete'").fetchall()
        return {str(row["episode_id"]) for row in rows}

    def progress_snapshot(self, total: int) -> dict[str, Any]:
        rows = self.connection.execute(
            "SELECT status,COUNT(*) n FROM episodes GROUP BY status"
        ).fetchall()
        statuses = {str(row["status"]): int(row["n"]) for row in rows}
        completed = statuses.get("complete", 0)
        successes = int(self.connection.execute(
            "SELECT COUNT(*) n FROM episodes WHERE status='complete' AND success=1"
        ).fetchone()["n"])
        bootstrap = int(self.connection.execute(
            """SELECT COUNT(DISTINCT episode_id) n FROM evaluations
               WHERE turn_index=1 AND json_extract(record_json,'$.verified_success')=1"""
        ).fetchone()["n"])
        adaptive = int(self.connection.execute(
            """SELECT COUNT(DISTINCT episode_id) n FROM evaluations
               WHERE turn_index>=2 AND json_extract(record_json,'$.verified_success')=1"""
        ).fetchone()["n"])
        queries = int(self.connection.execute("SELECT COUNT(*) n FROM transcripts").fetchone()["n"])
        verified = int(self.connection.execute(
            """SELECT COUNT(*) n FROM evaluations
               WHERE json_extract(record_json,'$.verification_status')='fully_verified'"""
        ).fetchone()["n"])
        screened = int(self.connection.execute(
            """SELECT COUNT(*) n FROM evaluations
               WHERE json_extract(record_json,'$.verification_status')='screened_only'"""
        ).fetchone()["n"])
        failures = int(self.connection.execute("SELECT COUNT(*) n FROM failures").fetchone()["n"])
        return {
            "completed": completed, "total": total, "successes": successes,
            "bootstrap_successes": bootstrap, "adaptive_successes": adaptive,
            "queries": queries, "fully_verified_turns": verified, "screened_turns": screened,
            "pending_verification": statuses.get("pending_verification", 0),
            "ready": statuses.get("ready", 0), "failures": failures,
        }

    def backup(self, destination: str | Path) -> None:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(destination) as target:
            self.connection.backup(target)


def sync_checkpoint_to_gcs(local_path: str | Path, uri: str | None) -> None:
    if not uri:
        return
    if not uri.startswith("gs://"):
        raise ValueError("gcs_checkpoint_uri must start with gs://")
    from google.cloud import storage

    bucket_name, object_name = uri[5:].split("/", 1)
    storage.Client().bucket(bucket_name).blob(object_name).upload_from_filename(str(local_path))
