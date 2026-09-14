from pathlib import Path
import pandas as pd
from chembreak12.dataset import load_task_bank
from chembreak12.integrity import verify_lock
ROOT=Path(__file__).resolve().parents[1]
def test_lock():
    s=ROOT/'data/final_task_bank.csv'; m=ROOT/'data/CB12_partition_manifest_v1.csv'; l=ROOT/'data/CB12_partition_lock_v1.json'
    report=verify_lock(source_path=s, frame=load_task_bank(s), manifest_path=m, manifest=pd.read_csv(m), lock_path=l)
    assert report['ok']
