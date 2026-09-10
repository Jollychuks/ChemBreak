from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cloud_notebook_has_independent_conditions_and_valid_syntax():
    path = ROOT / "notebooks/chembreak8_Cloud_Notebook.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    text = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert 'run_one("C3_ADAPTIVE_MDP", "ChemDFM")' in text
    assert 'run_one("C3_ADAPTIVE_MDP", "ChemLLM")' in text
    assert 'run_one("C3_ADAPTIVE_MDP", "LlaSMol")' in text
    assert "C0_DIRECT" in text
    assert "C1_REPEATED_SINGLE" in text
    assert "C2_FIXED_MULTI" in text
    assert "python -m venv" not in text
    assert "torch==" not in text
    assert "import json" in text
    assert "for version in range(1, 8)" in text
    assert 'PROJECT_ID.startswith("REPLACE_")' in text
    assert 'REPO_URL.startswith(("https://github.com/", "git@github.com:"))' in text
    assert "gemini-2.5-flash" not in text or "config" in text
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"cell_{index}")


def test_no_previous_package_dependency():
    for path in (ROOT / "src/chembreak8").glob("*.py"):
        text = path.read_text(encoding="utf-8").casefold()
        for version in range(1, 8):
            assert f"chembreak{version}" not in text


def test_source_roles_exclude_the_unreliable_gpt_oss_path():
    config_text = (ROOT / "configs/config.development.yaml").read_text(encoding="utf-8").casefold()
    assert "gpt-oss" not in config_text
    assert "gemini-2.5-flash" in config_text
    assert "llama-4-maverick" in config_text
