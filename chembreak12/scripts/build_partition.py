from pathlib import Path
import argparse
import pandas as pd
from chembreak12.dataset import load_task_bank
from chembreak12.partition import build_manifest, write_manifest, balance_report
from chembreak12.integrity import write_lock, verify_manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project-root', default='.')
    p.add_argument('--verify-only', action='store_true')
    a=p.parse_args()
    root=Path(a.project_root).resolve()
    source=root/'data/final_task_bank.csv'
    manifest_path=root/'data/CB12_partition_manifest_v1.csv'
    lock_path=root/'data/CB12_partition_lock_v1.json'
    audit_path=root/'data/CB12_partition_audit_v1.json'
    frame=load_task_bank(source)
    reproduced=build_manifest(frame)
    if manifest_path.exists():
        existing=pd.read_csv(manifest_path)
        verify_manifest(frame, existing)
        same=existing.fillna('').astype(str).equals(reproduced.fillna('').astype(str))
        if not same:
            raise RuntimeError('Existing manifest does not match deterministic CB12 reproduction.')
        print('Partition reproduced exactly: True')
    elif a.verify_only:
        raise FileNotFoundError(manifest_path)
    else:
        write_manifest(reproduced, manifest_path)
        existing=reproduced
        print('Partition created:', manifest_path)
    write_lock(source_path=source, frame=frame, manifest_path=manifest_path, manifest=existing,
               lock_path=lock_path, audit_path=audit_path)
    print(existing['split'].value_counts().to_dict())

if __name__=='__main__': main()
