from pathlib import Path
import json
from chembreak14.dataset import load_task_bank, load_mini_manifest
from chembreak14.selection import build_mini24, verify_bundle
from chembreak14.constants import FROZEN_ASSIGNMENT_IDS, FROZEN_ASSIGNMENT_IDS_SHA256
ROOT=Path(__file__).resolve().parents[1]

def test_selection_reproduces_and_lock_metadata_matches():
    bank=load_task_bank(ROOT/'data/final_task_bank.csv'); locked=load_mini_manifest(ROOT/'data/CB14_mini24_manifest_v1.csv'); reproduced=build_mini24(bank)
    assert locked.fillna('').astype(str).equals(reproduced.fillna('').astype(str))
    assert tuple(locked.assignment_id.astype(str))==tuple(FROZEN_ASSIGNMENT_IDS)
    assert len(locked)==24 and locked.hd_id.nunique()==8 and locked.ot_id.nunique()==15 and locked.hc_id.nunique()==9
    assert locked.hc_id.value_counts().min()>=2
    report=verify_bundle(ROOT/'data/final_task_bank.csv',ROOT/'data/CB14_mini24_manifest_v1.csv',ROOT/'data/CB14_mini24_lock_v1.json')
    assert report['status']=='ok' and report['ot_coverage']==15
    lock=json.loads((ROOT/'data/CB14_mini24_lock_v1.json').read_text())
    assert lock['assignment_ids_sha256']==FROZEN_ASSIGNMENT_IDS_SHA256 and 'seed' not in lock
