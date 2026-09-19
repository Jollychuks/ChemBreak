from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]

def test_exactly_one_notebook_and_clean_execution_state():
    nbs=list((ROOT/'notebooks').glob('*.ipynb')); assert len(nbs)==1; nb=json.loads(nbs[0].read_text()); assert len(nb['cells'])>=10
    for c in nb['cells']:
        if c.get('cell_type')=='code':assert c.get('execution_count') is None and not c.get('outputs')

def test_notebook_has_no_openai_key_cell():
    text=(ROOT/'notebooks/chembreak21_Cloud_Notebook.ipynb').read_text(); assert 'OPENAI_API_KEY' not in text and 'gpt-5.6-sol' not in text

def test_notebook_import_fix_present():
    text=(ROOT/'notebooks/chembreak21_Cloud_Notebook.ipynb').read_text(); assert 'sys.path.insert' in text and 'import chembreak21' in text

def test_release_design_excludes_internal_sqlite():
    text=(ROOT/'notebooks/chembreak21_Cloud_Notebook.ipynb').read_text(); assert 'public_root' in text and 'release' in text and 'state.sqlite3' in text
