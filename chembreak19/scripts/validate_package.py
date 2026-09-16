import ast, hashlib, json, re, sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from chembreak19.selection import verify_bundle
from chembreak19.policy import QPolicy
from chembreak19.evidence import EvidenceMemory
from chembreak19.trajectory import TrajectoryMemory

selection=verify_bundle(ROOT/'data/final_task_bank.csv',ROOT/'data/CB19_mini24_manifest_v1.csv',ROOT/'data/CB19_mini24_lock_v1.json')
required=[
 'README.md','CHANGES.md','pyproject.toml','requirements.txt','requirements-cloud-ml.txt','requirements-dev.txt',
 'configs/config.cb19.yaml','data/final_task_bank.csv','data/CB19_mini24_manifest_v1.csv','data/CB19_mini24_lock_v1.json',
 'notebooks/chembreak19_Cloud_Notebook.ipynb','src/chembreak19/runner.py','src/chembreak19/checkpoint.py',
 'src/chembreak19/targets.py','src/chembreak19/providers.py','src/chembreak19/policy.py','src/chembreak19/evidence.py','src/chembreak19/trajectory.py','src/chembreak19/state.py',
]
missing=[p for p in required if not (ROOT/p).is_file()]
if missing:raise AssertionError(f'Missing required package files: {missing}')
notebooks=list(ROOT.rglob('*.ipynb')); assert notebooks==[ROOT/'notebooks/chembreak19_Cloud_Notebook.ipynb'],f'Expected exactly one notebook, found {notebooks}'

cfg=yaml.safe_load((ROOT/'configs/config.cb19.yaml').read_text()); assert cfg['run']['namespace']=='CB19'
assert 'version="19.0.0"' in (ROOT/'pyproject.toml').read_text(); assert '__version__ = "19.0.0"' in (ROOT/'src/chembreak19/__init__.py').read_text()
assert cfg['run']['experiment_revision']=='CB19_TRAJECTORY_MDP_MINI24_V1'; assert [float(x) for x in cfg['experiment']['epoch_epsilons']]==[0.30,0.20,0.15]; assert int(cfg['experiment']['task_count'])==24
assert cfg['roles']['attack_llm']['model']=='gpt-5.6-sol'; assert cfg['roles']['judge_llm']['model']=='gemini-2.5-flash'
assert QPolicy.SCHEMA_VERSION==6; assert EvidenceMemory.SCHEMA_VERSION==1; assert TrajectoryMemory.SCHEMA_VERSION==1
for key in ['reliability_weight','support_weight','reward_weight','q_weight','training_artifact_path']:assert key in cfg['evidence']
for key in ['reliability_weight','support_weight','reward_weight','q_weight','training_artifact_path','freeze_snapshot_path','rankings_artifact_path']:assert key in cfg['trajectory']
assert cfg['evidence']['replay_during_learning'] is False and cfg['trajectory']['replay_during_learning'] is True

nb=json.loads(notebooks[0].read_text()); text='\n'.join(''.join(c.get('source',[])) for c in nb['cells'])
for term in ['PROJECT_ID','REPO_URL','PROJECT_SUBDIR','CB19_TRAJECTORY_MDP_MINI24_V1','chembreak19_storage','GOOGLE_CLOUD_PROJECT','OPENAI_API_KEY','gpt-5.6-sol','gemini-2.5-flash','LIVE_PROGRESS','running_ASR','0.30 → 0.20 → 0.15','successful trajectory memory','ASR@1','ASR@4','trajectory_rankings.csv']:
    assert term in text,f'Notebook missing {term}'
cell_text=[''.join(c.get('source',[])) for c in nb['cells']]
install_index=next(i for i,x in enumerate(cell_text) if 'Install the CB19 dependency stack' in x); verify_index=next(i for i,x in enumerate(cell_text) if 'Verify the fixed 24-task CB19 panel' in x); assert install_index<verify_index
for i,cell in enumerate(nb.get('cells',[]),1):
    if cell.get('cell_type')=='code':compile(''.join(cell.get('source',[])),f'notebook-cell-{i}','exec'); assert cell.get('execution_count') is None and not cell.get('outputs')
for p in list((ROOT/'src').rglob('*.py'))+list((ROOT/'scripts').rglob('*.py'))+list((ROOT/'tests').rglob('*.py')):ast.parse(p.read_text(),filename=str(p))

forbidden=[]
for i in range(1,19):
    forbidden.extend([f'chembreak{i}_',f'chembreak{i}.',f'chembreak{i}-',f'chembreak{i}/',f'cb{i}_',f'cb{i}.',f'cb{i}-',f'cb{i}/',f'/content/chembreak{i}_storage'])
for p in ROOT.rglob('*'):
    if p.is_file() and any(tok in p.name.lower() for tok in forbidden):raise AssertionError(f'Cross-version file found: {p.relative_to(ROOT)}')
for base in [ROOT/'src',ROOT/'scripts',ROOT/'configs',ROOT/'docs',ROOT/'notebooks']:
    for p in base.rglob('*'):
        if p.is_file():
            low=p.read_text(errors='ignore').lower()
            if any(tok in low for tok in forbidden):raise AssertionError(f'Prior-version runtime name found in {p.relative_to(ROOT)}')

manifest_path=ROOT/'PACKAGE_MANIFEST.sha256'
if manifest_path.exists():
    manifest={}
    for line in manifest_path.read_text().splitlines():
        if line.strip():digest,rel=line.split('  ',1); manifest[rel]=digest
    actual={str(p.relative_to(ROOT).as_posix()):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*') if p.is_file() and p.name!='PACKAGE_MANIFEST.sha256' and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts}
    if manifest!=actual:
        missing_from_manifest=sorted(set(actual)-set(manifest)); extra_in_manifest=sorted(set(manifest)-set(actual)); bad=sorted(k for k in set(actual)&set(manifest) if actual[k]!=manifest[k])
        raise AssertionError(f'PACKAGE_MANIFEST mismatch: missing={missing_from_manifest}, extra={extra_in_manifest}, bad_hash={bad}')
print(json.dumps({'status':'ok','selection':selection,'notebook_cells':len(nb['cells']),'epsilon_schedule':[0.30,0.20,0.15],'policy_schema_version':QPolicy.SCHEMA_VERSION,'evidence_schema_version':EvidenceMemory.SCHEMA_VERSION,'trajectory_schema_version':TrajectoryMemory.SCHEMA_VERSION,'notebook_count':1},indent=2,sort_keys=True))
