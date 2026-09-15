from pathlib import Path
from chembreak13.dataset import load_task_bank, load_mini_manifest
from chembreak13.selection import build_mini24, verify_bundle
ROOT=Path(__file__).resolve().parents[1]
def test_selection_reproduces():
    bank=load_task_bank(ROOT/'data/final_task_bank.csv'); locked=load_mini_manifest(ROOT/'data/CB13_mini24_manifest_v1.csv'); reproduced=build_mini24(bank)
    assert locked.fillna('').astype(str).equals(reproduced.fillna('').astype(str))
    assert len(locked)==24 and locked.hd_id.nunique()==8 and locked.ot_id.nunique()==15 and locked.hc_id.nunique()==9
    assert locked.hc_id.value_counts().min()>=2
    assert verify_bundle(ROOT/'data/final_task_bank.csv',ROOT/'data/CB13_mini24_manifest_v1.csv',ROOT/'data/CB13_mini24_lock_v1.json')['status']=='ok'
