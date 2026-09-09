from pathlib import Path

from chembreak6.config import load_config, validate_config


ROOT = Path(__file__).resolve().parents[1]


def test_test_config_contract():
    config = load_config(ROOT / "configs" / "config.test.yaml")
    validate_config(config)
    assert config["experiment"]["task_count"] == 8
    assert config["experiment"]["target_query_budget"] == 5
    assert config["experiment"]["condition_query_budgets"]["C0_DIRECT"] == 1
    assert config["roles"]["safety_judge"]["structured_output_mode"] == "client_validated_json"


def test_inheritance():
    config = load_config(ROOT / "configs" / "config.pilot.yaml")
    validate_config(config)
    assert config["experiment"]["task_count"] == 40
    assert config["roles"]["planner"]["model"] == "gemini-3.1-pro-preview"
