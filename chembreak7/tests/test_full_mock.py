from __future__ import annotations

import sqlite3

import pandas as pd
from conftest import make_config

from chembreak7.runner import run_target, strict_completion_gate


def test_full_adaptive_mock_and_resume(tmp_path):
    config_path = make_config(tmp_path)
    run_dir = None
    for target in ("ChemDFM", "ChemLLM", "LlaSMol"):
        run_dir = run_target(config_path, target)
    assert run_dir is not None
    completion = strict_completion_gate(config_path)
    assert completion["episodes"] == 24
    with sqlite3.connect(run_dir / "state.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM episodes WHERE status='complete'").fetchone()[0] == 24
        assert connection.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0] == 120
        assert connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 120
        assert connection.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0] == 120
        assert connection.execute("SELECT COUNT(*) FROM failures").fetchone()[0] == 0
        actor_calls = connection.execute(
            "SELECT COUNT(*) FROM api_calls WHERE json_extract(record_json,'$.role')='adaptive_actor' AND json_extract(record_json,'$.status')='valid_json'"
        ).fetchone()[0]
        assert actor_calls == 96
        before = connection.execute("SELECT COUNT(*) FROM api_calls").fetchone()[0]
    run_target(config_path, "ChemDFM")
    with sqlite3.connect(run_dir / "state.sqlite3") as connection:
        after = connection.execute("SELECT COUNT(*) FROM api_calls").fetchone()[0]
        assert after == before
    results = pd.read_csv(run_dir / "release/episode_results.csv")
    assert len(results) == 24
    assert not results.verified_success.any()
    metrics = pd.read_csv(run_dir / "release/adaptive_mdp_metrics.csv")
    assert set(metrics.target_id) == {"ChemDFM", "ChemLLM", "LlaSMol", "ALL_TARGETS"}

