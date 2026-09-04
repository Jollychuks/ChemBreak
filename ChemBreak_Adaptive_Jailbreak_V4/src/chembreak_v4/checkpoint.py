from __future__ import annotations

import json
import shutil
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
CREATE TABLE IF NOT EXISTS turns (
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

    def start_episode(self, episode_id: str, assignment_id: str, target_id: str, condition: str) -> None:
        self.connection.execute(
            """INSERT OR IGNORE INTO episodes(
                episode_id, assignment_id, target_id, condition, status, started_at
            ) VALUES (?, ?, ?, ?, 'running', ?)""",
            (episode_id, assignment_id, target_id, condition, utc_now()),
        )
        self.connection.commit()

    def existing_turns(self, episode_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT record_json FROM turns WHERE episode_id=? ORDER BY turn_index", (episode_id,)
        ).fetchall()
        return [json.loads(row["record_json"]) for row in rows]

    def save_turn(self, turn: TurnRecord) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO turns VALUES (?, ?, ?, ?)",
            (
                turn.episode_id,
                turn.turn_index,
                json.dumps(turn.to_dict(), ensure_ascii=False),
                utc_now(),
            ),
        )
        self.connection.execute(
            """UPDATE episodes SET queries_used=(SELECT COUNT(*) FROM turns WHERE episode_id=?),
               cumulative_reward=(SELECT COALESCE(SUM(json_extract(record_json, '$.reward')),0)
                                  FROM turns WHERE episode_id=?) WHERE episode_id=?""",
            (turn.episode_id, turn.episode_id, turn.episode_id),
        )
        self.connection.commit()

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

    def record_failure(self, episode_id: str | None, stage: str, error: Exception) -> None:
        self.connection.execute(
            """INSERT INTO failures(episode_id, stage, error_type, error_message, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (episode_id, stage, type(error).__name__, str(error)[:4000], utc_now()),
        )
        self.connection.commit()

    def completed_episode_ids(self) -> set[str]:
        rows = self.connection.execute(
            "SELECT episode_id FROM episodes WHERE status='complete'"
        ).fetchall()
        return {str(row["episode_id"]) for row in rows}

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
