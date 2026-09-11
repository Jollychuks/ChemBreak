from __future__ import annotations

import sqlite3

from conftest import make_config

from chembreak9.benchmark import load_and_validate_task_bank, to_task_records
from chembreak9.checkpoint import CheckpointStore
from chembreak9.conditions import run_episode
from chembreak9.config import load_config
from chembreak9.policy import AdaptiveQPolicy
from chembreak9.providers import RoleClients
from chembreak9.targets import MockTarget


def _setup(tmp_path):
    config = load_config(make_config(tmp_path))
    task = to_task_records(load_and_validate_task_bank(config["run"]["task_bank_path"]).iloc[:1])[0]
    store = CheckpointStore(tmp_path / "conditions.sqlite3", "sig", {})
    clients = RoleClients(config, None)
    policy = AdaptiveQPolicy(config["policy"], int(config["run"]["seed"]))
    return config, task, store, clients, policy


def test_four_conditions_have_distinct_query_behaviour(tmp_path):
    config, task, store, clients, policy = _setup(tmp_path)
    results = {}
    for condition in config["experiment"]["conditions"]:
        results[condition] = run_episode(
            episode_id=f"e-{condition}", task=task, target_id="ChemDFM",
            target=MockTarget("ChemDFM"), condition=condition, clients=clients,
            store=store, config=config, learned_policy=policy,
        )
    assert results["C0_DIRECT"]["queries_used"] == 1
    assert results["C1_REPEATED_SINGLE"]["queries_used"] == 8
    assert results["C2_FIXED_MULTI"]["queries_used"] == 8
    assert results["C3_ADAPTIVE_MDP"]["queries_used"] == 8
    assert [row["action_id"] for row in store.turn_rows("e-C1_REPEATED_SINGLE")] == ["BOOTSTRAP_DIRECT"] + ["REPEAT_DIRECT"] * 7
    assert len({row["action_id"] for row in store.turn_rows("e-C2_FIXED_MULTI")[1:]}) == 7
    store.close()


def test_fixed_multi_prompts_are_reused_across_targets(tmp_path):
    config, task, store, clients, policy = _setup(tmp_path)
    for target_id in ("ChemDFM", "ChemLLM"):
        run_episode(
            episode_id=f"e-{target_id}", task=task, target_id=target_id,
            target=MockTarget(target_id), condition="C2_FIXED_MULTI", clients=clients,
            store=store, config=config, learned_policy=policy,
        )
    first, second = store.turn_rows("e-ChemDFM"), store.turn_rows("e-ChemLLM")
    assert [row["attack_prompt"] for row in first] == [row["attack_prompt"] for row in second]
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM fixed_prompts").fetchone()[0] == 7
        assert connection.execute(
            "SELECT COUNT(*) FROM api_calls WHERE json_extract(record_json,'$.role')='adaptive_actor'"
        ).fetchone()[0] == 7
    store.close()
