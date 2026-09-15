import ast, hashlib, json, re, sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from chembreak16.selection import verify_bundle
from chembreak16.policy import QPolicy

selection=verify_bundle(ROOT/'data/final_task_bank.csv',ROOT/'data/CB16_mini24_manifest_v1.csv',ROOT/'data/CB16_mini24_lock_v1.json')
required=[
 'README.md','CHANGES.md','pyproject.toml','requirements.txt','requirements-cloud-ml.txt','requirements-dev.txt',
 'configs/config.cb16.yaml','data/final_task_bank.csv','data/CB16_mini24_manifest_v1.csv','data/CB16_mini24_lock_v1.json',
 'notebooks/chembreak16_Cloud_Notebook.ipynb','src/chembreak16/runner.py','src/chembreak16/checkpoint.py',
 'src/chembreak16/targets.py','src/chembreak16/providers.py','src/chembreak16/policy.py','src/chembreak16/state.py',
]
missing=[p for p in required if not (ROOT/p).is_file()]
if missing: raise AssertionError(f'Missing required package files: {missing}')

notebooks=list(ROOT.rglob('*.ipynb'))
assert notebooks==[ROOT/'notebooks/chembreak16_Cloud_Notebook.ipynb'],f'Expected exactly one notebook, found {notebooks}'

cfg=yaml.safe_load((ROOT/'configs/config.cb16.yaml').read_text())
assert cfg['run']['namespace']=='CB16'

assert 'version="16.1.1"' in (ROOT/'pyproject.toml').read_text()
assert '__version__ = "16.1.1"' in (ROOT/'src/chembreak16/__init__.py').read_text()
assert "'package_version':'16.1.1'" in (ROOT/'src/chembreak16/runner.py').read_text()
assert cfg['run']['experiment_revision']=='CB16_HIER_MDP_MINI24_V1'
assert [float(x) for x in cfg['experiment']['epoch_epsilons']]==[0.30,0.20,0.15]
assert int(cfg['experiment']['task_count'])==24
for key in ['global_weight','hc_weight','hd_weight','ot_weight','task_weight','global_learning_rate','context_learning_rate','task_learning_rate']:
    assert key in cfg['policy']
assert QPolicy.SCHEMA_VERSION==3

nb=json.loads(notebooks[0].read_text()); text='\n'.join(''.join(c.get('source',[])) for c in nb['cells'])
for term in ['PROJECT_ID','REPO_URL','PROJECT_SUBDIR','CB16_HIER_MDP_MINI24_V1','chembreak16_storage','GOOGLE_CLOUD_PROJECT','LIVE_PROGRESS','running_ASR','0.30 → 0.20 → 0.15','mode=cold_start','Qhc','Qhd','Qot','policy_diagnostics.csv','policy_support_summary.json']:
    assert term in text,f'Notebook missing {term}'
cell_text=[''.join(c.get('source',[])) for c in nb['cells']]
install_index=next(i for i,x in enumerate(cell_text) if 'Install the CB16 dependency stack' in x)
verify_index=next(i for i,x in enumerate(cell_text) if 'Verify the fixed 24-task CB16 panel' in x)
assert install_index < verify_index, 'Dependency install must precede pandas-based bundle verification'

for i,cell in enumerate(nb.get('cells',[]),1):
    if cell.get('cell_type')=='code':
        compile(''.join(cell.get('source',[])),f'notebook-cell-{i}','exec')
        assert cell.get('execution_count') is None and not cell.get('outputs'),f'Notebook cell {i} contains saved execution state'

for p in list((ROOT/'src').rglob('*.py'))+list((ROOT/'scripts').rglob('*.py'))+list((ROOT/'tests').rglob('*.py')):
    ast.parse(p.read_text(),filename=str(p))

# No runtime/data files from another ChemBreak package are allowed in this package.
old_path_pattern=re.compile(r'(?:^|/)(?:chembreak(?:[1-9]|1[0-5])|cb(?:[1-9]|1[0-5])_)(?!\d)',re.I)
for p in ROOT.rglob('*'):
    rel=p.relative_to(ROOT).as_posix()
    if old_path_pattern.search(rel):
        raise AssertionError(f'Cross-version file/path found: {rel}')

# Prior-version runtime names must not appear in executable/config/docs/notebook content.
forbidden=[]
for i in range(1,16):
    forbidden.extend([f'chembreak{i}_',f'chembreak{i}.',f'chembreak{i}-',f'chembreak{i}/',f'cb{i}_',f'cb{i}.',f'cb{i}-',f'cb{i}/',f'/content/chembreak{i}_storage'])
for base in [ROOT/'src',ROOT/'scripts',ROOT/'configs',ROOT/'docs',ROOT/'notebooks']:
    for p in base.rglob('*'):
        if p.is_file():
            low=p.read_text(errors='ignore').lower()
            if any(tok in low for tok in forbidden):
                raise AssertionError(f'Prior-version runtime name found in {p.relative_to(ROOT)}')

manifest_path=ROOT/'PACKAGE_MANIFEST.sha256'
if manifest_path.exists():
    manifest={}
    for line in manifest_path.read_text().splitlines():
        if line.strip():
            digest,rel=line.split('  ',1); manifest[rel]=digest
    actual={str(p.relative_to(ROOT).as_posix()):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*') if p.is_file() and p.name!='PACKAGE_MANIFEST.sha256' and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts}
    if manifest!=actual:
        missing_from_manifest=sorted(set(actual)-set(manifest)); extra_in_manifest=sorted(set(manifest)-set(actual)); bad=sorted(k for k in set(actual)&set(manifest) if actual[k]!=manifest[k])
        raise AssertionError(f'PACKAGE_MANIFEST mismatch: missing={missing_from_manifest}, extra={extra_in_manifest}, bad_hash={bad}')

print(json.dumps({'status':'ok','selection':selection,'notebook_cells':len(nb['cells']),'epsilon_schedule':[0.30,0.20,0.15],'policy_schema_version':QPolicy.SCHEMA_VERSION,'notebook_count':1},indent=2,sort_keys=True))
