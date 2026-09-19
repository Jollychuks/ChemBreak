from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
def test_exactly_one_notebook():assert len(list((ROOT/'notebooks').glob('*.ipynb')))==1
def test_notebook_clean_and_controls_present():
    nb=json.loads(next((ROOT/'notebooks').glob('*.ipynb')).read_text()); assert all(not c.get('outputs') for c in nb['cells'] if c.get('cell_type')=='code'); text='\n'.join(''.join(c.get('source',[])) for c in nb['cells']); assert 'PROJECT_SUBDIR      = "chembreak20"' in text and 'TARGETS             = ["ChemDFM", "ChemLLM"]' in text
def test_no_old_runtime_namespace_strings():
    bad=[]
    tokens=('chembreak'+'19_storage','config.'+'cb19','cb19_'+'trajectory_mdp')
    for p in ROOT.rglob('*'):
        if p.is_file() and p.suffix in {'.py','.yaml','.md','.json','.toml'}:
            t=p.read_text(errors='ignore').lower()
            for token in tokens:
                if token in t:bad.append((str(p),token))
    assert not bad,bad
