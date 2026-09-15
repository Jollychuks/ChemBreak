import ast, hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from chembreak14.selection import verify_bundle

selection=verify_bundle(ROOT/'data/final_task_bank.csv',ROOT/'data/CB14_mini24_manifest_v1.csv',ROOT/'data/CB14_mini24_lock_v1.json')

required=[
 'README.md','CHANGES.md','pyproject.toml','requirements.txt','requirements-cloud-ml.txt','requirements-dev.txt',
 'configs/config.cb14.yaml','data/final_task_bank.csv','data/CB14_mini24_manifest_v1.csv','data/CB14_mini24_lock_v1.json',
 'chembreak14_Cloud_Notebook.ipynb','notebooks/chembreak14_Cloud_Notebook.ipynb',
 'src/chembreak14/runner.py','src/chembreak14/checkpoint.py','src/chembreak14/targets.py','src/chembreak14/providers.py',
]
missing=[p for p in required if not (ROOT/p).is_file()]
if missing: raise AssertionError(f'Missing required package files: {missing}')

root_nb=ROOT/'chembreak14_Cloud_Notebook.ipynb'; nested_nb=ROOT/'notebooks/chembreak14_Cloud_Notebook.ipynb'
assert root_nb.read_bytes()==nested_nb.read_bytes(), 'Root and notebooks/ cloud notebooks differ'
nb=json.loads(root_nb.read_text())
text=root_nb.read_text()
for term in ['PROJECT_ID','REPO_URL','PROJECT_SUBDIR','CB14_MDP_MINI24_V1','chembreak14_storage','GOOGLE_CLOUD_PROJECT','LIVE_PROGRESS','running_ASR']:
    assert term in text, f'Notebook missing {term}'
for forbidden in ['/content/chembreak13_storage','/content/chembreak12_storage','/content/chembreak11_storage','PROJECT_SUBDIR      = "chembreak13"','PROJECT_SUBDIR      = "chembreak12"']:
    assert forbidden not in text, f'Old-version runtime reference found: {forbidden}'
for i,cell in enumerate(nb.get('cells',[]),1):
    if cell.get('cell_type')=='code':
        compile(''.join(cell.get('source',[])),f'notebook-cell-{i}','exec')
        assert not cell.get('outputs'), f'Notebook cell {i} contains saved output'

# Parse every Python file so syntax failures are caught even without importing optional cloud dependencies.
for p in list((ROOT/'src').rglob('*.py'))+list((ROOT/'scripts').rglob('*.py'))+list((ROOT/'tests').rglob('*.py')):
    ast.parse(p.read_text(),filename=str(p))

# Verify the release manifest itself covers every package file except itself.
manifest_path=ROOT/'PACKAGE_MANIFEST.sha256'
manifest={}
for line in manifest_path.read_text().splitlines():
    if not line.strip(): continue
    digest,rel=line.split('  ',1); manifest[rel]=digest
actual={str(p.relative_to(ROOT).as_posix()):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*') if p.is_file() and p.name!='PACKAGE_MANIFEST.sha256' and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts}
if manifest!=actual:
    missing_from_manifest=sorted(set(actual)-set(manifest)); extra_in_manifest=sorted(set(manifest)-set(actual)); bad=sorted(k for k in set(actual)&set(manifest) if actual[k]!=manifest[k])
    raise AssertionError(f'PACKAGE_MANIFEST mismatch: missing={missing_from_manifest}, extra={extra_in_manifest}, bad_hash={bad}')

print(json.dumps({'status':'ok','selection':selection,'files_verified':len(actual),'notebook_cells':len(nb['cells'])},indent=2,sort_keys=True))
