from __future__ import annotations

import sqlite3

import pandas as pd
from conftest import make_config

from chembreak8.config import PHASE_COUNTS, load_config
from chembreak8.runner import run_target, strict_completion_gate


def test_full_adaptive_mock_and_resume(tmp_path):
    config_path = make_config(tmp_path)
    run_dir = None
    for target in ("ChemDFM", "ChemLLM", "LlaSMol"):
        run_dir = run_target(config_path, target)
    assert run_dir is not None
    config = load_config(config_path)
    episodes = PHASE_COUNTS["development"] * 3
    budget = int(config["experiment"]["target_query_budget"])
    pool = int(config["policy"]["candidate_pool_size"])
    completion = strict_completion_gate(config_path, "C3_ADAPTIVE_MDP")
    assert completion["episodes"] == episodes
    with sqlite3.connect(run_dir / "state.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM episodes WHERE status='complete'").fetchone()[0] == episodes
        assert connection.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0] == episodes * budget
        assert connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == episodes * budget
        assert connection.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0] == episodes * budget
        assert connection.execute("SELECT COUNT(*) FROM failures").fetchone()[0] == 0
        actor_calls = connection.execute(
            "SELECT COUNT(*) FROM api_calls WHERE json_extract(record_json,'$.role')='adaptive_actor' AND json_extract(record_json,'$.status')='valid_json'"
        ).fetchone()[0]
        assert actor_calls == episodes * (budget - 1) * pool
        before = connection.execute("SELECT COUNT(*) FROM api_calls").fetchone()[0]
    run_target(config_path, "ChemDFM")
    with sqlite3.connect(run_dir / "state.sqlite3") as connection:
        after = connection.execute("SELECT COUNT(*) FROM api_calls").fetchone()[0]
        assert after == before
    results = pd.read_csv(run_dir / "release/episode_results.csv")
    assert len(results) == episodes
    assert not results.verified_success.any()
    metrics = pd.read_csv(run_dir / "release/adaptive_mdp_metrics.csv")
    assert set(metrics.target_id) == {"ChemDFM", "ChemLLM", "LlaSMol", "ALL_TARGETS"}
    assert set(metrics.condition) == {"C3_ADAPTIVE_MDP"}
