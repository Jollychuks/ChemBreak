"""Static contract checks for the clean release. No model download or API calls."""
from pathlib import Path
import hashlib
import json
import yaml

from chembreak36.config import load_config, validate_config
from chembreak36.constants import NAMESPACE, PACKAGE_VERSION, SELECTION_PROTOCOL
from chembreak36.selection import verify_bundle

ROOT = Path(__file__).resolve().parents[1]


def test_release_labels_and_clean_manifest():
    assert NAMESPACE == "CB36"
    assert PACKAGE_VERSION == "36.0.0"
    assert SELECTION_PROTOCOL == "CB36_PROMPTS28_V1"
    config = load_config(ROOT / "configs" / "config.cb36.yaml")
    assert validate_config(config) is True
    assert config["run"]["namespace"] == "CB36"
    assert verify_bundle(config["run"]["prompts_path"], config["run"]["manifest_path"], config["run"]["lock_path"])["tasks"] == 28
    assert hashlib.sha256((ROOT / 'data' / 'prompts.csv').read_bytes()).hexdigest() == '80c10ea78cf859174e5eb83d24c1bfe49b73b5adeab8b7b59c3045cb201c8434'
    ids = json.loads((ROOT / 'data' / 'CB36_prompts28_lock_v1.json').read_text())['assignment_ids']
    assert ids == [f'CB36P-{i:04d}' for i in range(1,29)]


def test_local_judge_and_notebook_paths_are_consistent():
    config = yaml.safe_load((ROOT / 'configs' / 'config.cb36.yaml').read_text())
    judge = config['roles']['chcs_judge_llm']
    assert judge['provider'] == 'local_llamacpp_gguf'
    assert judge['quantization'] == 'Q4_K_M'
    assert judge['model_path'].startswith('/content/chembreak36_storage/models/')
    assert judge['api_base_url'] == 'http://127.0.0.1:18036/v1'
    assert judge['enable_thinking'] is True
    assert judge['reasoning_budget_tokens'] == 2048
    assert judge['max_output_tokens'] == 4096
    nb = json.loads((ROOT / 'notebooks' / 'chembreak36_Cloud_Notebook.ipynb').read_text())
    sources = '\n'.join(''.join(c['source']) for c in nb['cells'])
    assert '--alias' in sources
    assert "'--reasoning', 'on'" in sources
    assert "'--reasoning-budget', '2048'" in sources
    assert "'--reasoning-format', 'deepseek'" in sources
    assert 'Thinking-mode probe PASSED' in sources
    assert 'Structured CHCS judge probe PASSED' in sources
    assert '--chat-template-kwargs' not in sources
    assert 'ChemBreak36_Standalone_Run_Package_v36.0.0.zip' in sources
    assert "from chembreak36.runner import ChemBreak36Runner" in sources


def test_no_stale_namespace_files_or_contents():
    # Use split literals so old-version strings are absent even in the tests.
    leftovers = ("chembreak" + "34", "cb" + "34", "chembreak" + "35", "cb" + "35")
    for file in ROOT.rglob('*'):
        if not file.is_file() or '__pycache__' in file.parts or '.pytest_cache' in file.parts:
            continue
        assert all(x not in str(file).casefold() for x in leftovers)
        assert all(x not in file.read_text().casefold() for x in leftovers)
