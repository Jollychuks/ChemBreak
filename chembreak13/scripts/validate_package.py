import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import json
from chembreak13.selection import verify_bundle
root=Path(__file__).resolve().parents[1]
r=verify_bundle(root/'data/final_task_bank.csv',root/'data/CB13_mini24_manifest_v1.csv',root/'data/CB13_mini24_lock_v1.json')
nb=(root/'notebooks/chembreak13_Cloud_Notebook.ipynb').read_text()
for term in ['PROJECT_ID','REPO_URL','PROJECT_SUBDIR','chembreak13_storage','GOOGLE_CLOUD_PROJECT']:
    assert term in nb, f'Notebook missing {term}'
for forbidden in ['/content/chembreak12_storage','/content/chembreak11_storage','PROJECT_SUBDIR      = "chembreak12"']:
    assert forbidden not in nb, f'Old-version runtime reference found: {forbidden}'
print(json.dumps({'status':'ok','selection':r},indent=2))
