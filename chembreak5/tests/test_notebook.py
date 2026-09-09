import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cloud_notebook_preserves_system_torch_stack():
    notebook = json.loads(
        (ROOT / "notebooks" / "chembreak5_Cloud_Notebook.ipynb").read_text(encoding="utf-8")
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
    assert "Using the existing complete" in source
    assert "The existing checkout has local changes" not in source
    assert "CONFIRMED JAILBREAKS FOUND" in source
    assert "SHOW_PRIVATE_OUTPUTS = False" in source
    assert "chembreak{version}_storage" in source
    assert "chembreak5_repo" in source
    assert "transformers==4.40.2" in source
    assert "probe_tokenizers=True" in source
    assert "run_coverage.csv" in source
    assert "import importlib\nimport json\nimport site" in source
