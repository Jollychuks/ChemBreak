import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cloud_notebook_preserves_system_torch_stack():
    notebook = json.loads(
        (ROOT / "notebooks" / "chembreak3_Cloud_Notebook.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert '"-m", "venv"' not in source
    assert '"--no-deps"' in source
    assert '"--upgrade"' not in source
    assert '"torch": "torch' not in source
    assert '"torchvision": "torchvision' not in source
    assert "Torch must come from the Notebook Enterprise image" in source
