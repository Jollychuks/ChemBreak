from __future__ import annotations

import sqlite3

from conftest import make_config

from chembreak7.runner import _episode_id, _initialize, recover_pending, run_target
from chembreak7.schema import TurnRecord


def test_recovery_observes_and_verifies_saved_target_response_without_requery(tmp_path):
    config_path = make_config(tmp_path)
    context = _initialize(config_path)
    task = context.tasks[0]
    episode_id = _episode_id(context.signature, "development", "ChemDFM", task.assignment_id)
    context.store.start_episode(episode_id, task.assignment_id, "ChemDFM")
    context.store.save_transcript(TurnRecord(
        episode_id, task.assignment_id, "ChemDFM", 1, "BOOTSTRAP_DIRECT",
        "Frozen original benchmark prompt. No actor call.", task.benchmark_prompt,
        "I cannot help with harmful chemical instructions.",
    ))
    context.store.close()

    recovery = recover_pending(config_path, "ChemDFM")
    assert recovery["attempted"] == 1
    assert recovery["recovered"] == 1
    with sqlite3.connect(recovery["run_dir"] + "/state.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM transcripts WHERE episode_id=?", (episode_id,)).fetchone()[0] == 1
        assert connection.execute("SELECT status FROM episodes WHERE episode_id=?", (episode_id,)).fetchone()[0] == "ready"

    run_dir = run_target(config_path, "ChemDFM")
    with sqlite3.connect(run_dir / "state.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM transcripts WHERE episode_id=?", (episode_id,)).fetchone()[0] == 5
        assert connection.execute("SELECT COUNT(*) FROM observations WHERE episode_id=?", (episode_id,)).fetchone()[0] == 5
        assert connection.execute("SELECT COUNT(*) FROM evaluations WHERE episode_id=?", (episode_id,)).fetchone()[0] == 5

