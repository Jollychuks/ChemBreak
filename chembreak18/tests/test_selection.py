from pathlib import Path
from chembreak18.dataset import load_task_bank, load_mini_manifest, selected_tasks
from chembreak18.selection import build_mini24, verify_bundle
ROOT=Path(__file__).resolve().parents[1]

def test_fixed_panel_reproduces_exactly_and_has_expected_coverage():
    bank=load_task_bank(ROOT/'data/final_task_bank.csv'); locked=load_mini_manifest(ROOT/'data/CB18_mini24_manifest_v1.csv'); rebuilt=build_mini24(bank)
    assert locked.fillna('').astype(str).equals(rebuilt.fillna('').astype(str))
    selected=selected_tasks(ROOT/'data/final_task_bank.csv',ROOT/'data/CB18_mini24_manifest_v1.csv')
    assert len(selected)==24 and not selected.is_reserve.any()
    assert locked.hc_id.nunique()==9 and locked.hd_id.nunique()==8 and locked.ot_id.nunique()==15
    report=verify_bundle(ROOT/'data/final_task_bank.csv',ROOT/'data/CB18_mini24_manifest_v1.csv',ROOT/'data/CB18_mini24_lock_v1.json')
    assert report['status']=='ok' and report['tasks']==24
