from pathlib import Path
import json,sys,yaml
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from chembreak16.selection import verify_bundle
from chembreak16.config import validate_config
selection=verify_bundle(ROOT/'data/final_task_bank.csv',ROOT/'data/CB12_partition_manifest_v1.csv',ROOT/'data/CB12_partition_lock_v1.json',ROOT/'data/CB16_train24_manifest_v1.csv',ROOT/'data/CB16_holdout12_manifest_v1.csv',ROOT/'data/CB16_selection_lock_v1.json')
required=['configs/config.cb16.yaml','data/final_task_bank.csv','data/CB12_partition_manifest_v1.csv','data/CB12_partition_lock_v1.json','data/CB16_train24_manifest_v1.csv','data/CB16_holdout12_manifest_v1.csv','data/CB16_selection_lock_v1.json','chembreak16_Cloud_Notebook.ipynb','notebooks/chembreak16_Cloud_Notebook.ipynb','src/chembreak16/runner.py','src/chembreak16/policy.py','src/chembreak16/state.py']
missing=[x for x in required if not (ROOT/x).exists()]
if missing: raise RuntimeError(f'Missing required files: {missing}')
cfg=yaml.safe_load((ROOT/'configs/config.cb16.yaml').read_text()); validate_config(cfg)
root_nb=ROOT/'chembreak16_Cloud_Notebook.ipynb'; nested_nb=ROOT/'notebooks/chembreak16_Cloud_Notebook.ipynb'
if root_nb.read_bytes()!=nested_nb.read_bytes(): raise RuntimeError('Notebook copies differ')
nb=json.loads(root_nb.read_text()); text="\n".join("".join(c.get("source",[])) for c in nb["cells"])
for term in ['PROJECT_ID','REPO_URL','PROJECT_SUBDIR','CB16_HIER_MDP_TRAIN24_TEST12_V1','chembreak16_storage','GOOGLE_CLOUD_PROJECT','LIVE_PROGRESS','running_ASR','0.30 → 0.20 → 0.15','mode=cold_start','Qhc','Qhd','Qot','holdout','policy_diagnostics.csv']:
    if term not in text: raise RuntimeError(f'Notebook missing {term}')
for old in ['chembreak15_storage','chembreak14_storage','CB16_MDP_MINI24_V1']:
    if old in text: raise RuntimeError(f'Notebook contains stale term {old}')
print(json.dumps({'status':'ok','selection':selection,'required_files':len(required),'experiment_revision':cfg['run']['experiment_revision']},indent=2,sort_keys=True))
