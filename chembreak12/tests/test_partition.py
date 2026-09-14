from pathlib import Path
import pandas as pd
from chembreak12.dataset import load_task_bank
from chembreak12.partition import build_manifest
from chembreak12.integrity import verify_manifest

ROOT=Path(__file__).resolve().parents[1]
def test_partition_exact_and_deterministic():
    frame=load_task_bank(ROOT/'data/final_task_bank.csv')
    locked=pd.read_csv(ROOT/'data/CB12_partition_manifest_v1.csv')
    verify_manifest(frame, locked)
    rebuilt=build_manifest(frame.sample(frac=1, random_state=77).reset_index(drop=True))
    assert locked.fillna('').astype(str).equals(rebuilt.fillna('').astype(str))
    assert locked['split'].value_counts().to_dict()=={'Train':241,'Reserve':59,'Test1':50,'Test2':50,'Test3':50,'Test4':50}
