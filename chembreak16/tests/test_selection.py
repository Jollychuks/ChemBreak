from pathlib import Path
from chembreak16.dataset import load_task_bank, load_partition_manifest, load_train_manifest, load_holdout_manifest
from chembreak16.selection import build_train24, build_holdout12, verify_bundle
ROOT=Path(__file__).resolve().parents[1]

def test_train_holdout_are_fixed_disjoint_and_follow_cb12_firewall():
    bank=load_task_bank(ROOT/'data/final_task_bank.csv'); part=load_partition_manifest(ROOT/'data/CB12_partition_manifest_v1.csv'); train=load_train_manifest(ROOT/'data/CB16_train24_manifest_v1.csv'); hold=load_holdout_manifest(ROOT/'data/CB16_holdout12_manifest_v1.csv')
    assert train.fillna('').astype(str).equals(build_train24(bank,part).fillna('').astype(str)); assert hold.fillna('').astype(str).equals(build_holdout12(bank,part).fillna('').astype(str))
    assert not(set(train.assignment_id)&set(hold.assignment_id)); assert set(train.source_split)=={'Train'}; assert set(hold.source_split)=={'Test1'}
    assert train.hc_id.nunique()==9 and train.hd_id.nunique()==8 and train.ot_id.nunique()==15
    assert hold.hc_id.nunique()==9 and hold.hd_id.nunique()==8 and hold.ot_id.nunique()==12
    report=verify_bundle(ROOT/'data/final_task_bank.csv',ROOT/'data/CB12_partition_manifest_v1.csv',ROOT/'data/CB12_partition_lock_v1.json',ROOT/'data/CB16_train24_manifest_v1.csv',ROOT/'data/CB16_holdout12_manifest_v1.csv',ROOT/'data/CB16_selection_lock_v1.json'); assert report['status']=='ok' and report['overlap']==0
