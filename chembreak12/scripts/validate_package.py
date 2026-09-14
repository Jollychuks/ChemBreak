from pathlib import Path
import pandas as pd
from chembreak12.dataset import load_task_bank
from chembreak12.partition import build_manifest
from chembreak12.integrity import verify_lock

root=Path(__file__).resolve().parents[1]
source=root/'data/final_task_bank.csv'
manifest_path=root/'data/CB12_partition_manifest_v1.csv'
lock_path=root/'data/CB12_partition_lock_v1.json'
frame=load_task_bank(source)
manifest=pd.read_csv(manifest_path)
report=verify_lock(source_path=source, frame=frame, manifest_path=manifest_path, manifest=manifest, lock_path=lock_path)
reproduced=build_manifest(frame)
assert manifest.fillna('').astype(str).equals(reproduced.fillna('').astype(str))
print('CB12 package validation: PASS')
print(report)
