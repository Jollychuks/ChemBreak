import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cloud_notebook_preserves_system_torch_stack():
    notebook = json.loads(
        (ROOT / "notebooks" / "chembreak6_Cloud_Notebook.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert '"-m", "venv"' not in source
    assert '"--no-deps"' in source
    assert '"torch": "torch' not in source
    assert '"torchvision": "torchvision' not in source
    assert "Torch must come from the Notebook Enterprise image" in source
    assert '"fetch", "origin", BRANCH' in source
    assert '"pull", "--ff-only"' in source
    assert "chembreak6_runtime_repo" in source
    assert "was left untouched" in source
    assert "SHOW_PRIVATE_OUTPUTS = False" in source
    assert "chembreak{version}_storage" in source
    assert "transformers==4.40.2" in source
    assert "probe_tokenizers=True" in source
    assert "run_coverage.csv" in source
    assert "import importlib\nimport json\nimport site" in source
    for condition in (
        "C0_DIRECT",
        "C1_REPEATED_SINGLE",
        "C2_FIXED_MULTI",
        "C3_ADAPTIVE_MDP",
    ):
        assert f'run_condition(runtime_path, "{condition}")' in source
    assert "recover_pending_judgments(runtime_path)" in source
    assert "runtime_dir = storage_root / \"runtime\"" in source
