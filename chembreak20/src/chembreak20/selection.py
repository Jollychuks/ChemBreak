from __future__ import annotations
import hashlib, json
from pathlib import Path
import pandas as pd
from .constants import NAMESPACE,SELECTION_PROTOCOL,FROZEN_ASSIGNMENT_IDS,FROZEN_ASSIGNMENT_IDS_SHA256
from .dataset import load_task_bank,load_mini_manifest,selected_tasks
from .utils import sha256_file

def _ids_sha(ids):return hashlib.sha256('\n'.join(map(str,ids)).encode()).hexdigest()

def build_mini24(bank: pd.DataFrame)->pd.DataFrame:
    order={a:i+1 for i,a in enumerate(FROZEN_ASSIGNMENT_IDS)}
    out=bank[bank.assignment_id.astype(str).isin(order)].copy()
    if len(out)!=24:
        missing=[a for a in FROZEN_ASSIGNMENT_IDS if a not in set(out.assignment_id.astype(str))]
        raise RuntimeError(f'Frozen CB20 panel incomplete; missing={missing}')
    out['selection_order']=out.assignment_id.astype(str).map(order)
    cols=['selection_order','assignment_id','matrix_id','hc_id','hc_category','hd_id','hazard_domain','ot_id','output_type','main_goal']
    return out[cols].sort_values('selection_order').reset_index(drop=True)

def verify_bundle(bank_path,manifest_path,lock_path):
    bank=load_task_bank(bank_path); locked=load_mini_manifest(manifest_path); reproduced=build_mini24(bank)
    if not locked.fillna('').astype(str).equals(reproduced.fillna('').astype(str)):raise RuntimeError('CB20 mini dataset reproduction failed')
    selected=selected_tasks(bank_path,manifest_path)
    if selected['is_reserve'].any():raise RuntimeError('Reserve row found in CB20 mini set')
    lock=json.loads(Path(lock_path).read_text())
    required={'namespace':NAMESPACE,'protocol':SELECTION_PROTOCOL,'task_count':24,'reserve_rows_allowed':False,'assignment_ids_sha256':FROZEN_ASSIGNMENT_IDS_SHA256}
    for k,v in required.items():
        if lock.get(k)!=v:raise RuntimeError(f'Lock metadata mismatch for {k}: {lock.get(k)!r} != {v!r}')
    if lock['source_task_bank_sha256']!=sha256_file(bank_path):raise RuntimeError('Source hash differs from lock')
    if lock['manifest_sha256']!=sha256_file(manifest_path):raise RuntimeError('Manifest hash differs from lock')
    if lock['assignment_ids']!=locked.assignment_id.astype(str).tolist():raise RuntimeError('Assignment list differs from lock')
    if _ids_sha(lock['assignment_ids'])!=FROZEN_ASSIGNMENT_IDS_SHA256:raise RuntimeError('Frozen assignment ID hash differs from lock')
    hc=locked.hc_id.value_counts().sort_index().to_dict(); hd=locked.hd_id.value_counts().sort_index().to_dict(); ot=locked.ot_id.value_counts().sort_index().to_dict()
    if len(hc)!=9 or len(hd)!=8 or len(ot)!=15:raise RuntimeError('CB20 taxonomy coverage invariant failed')
    return {'status':'ok','tasks':24,'hc_counts':hc,'hd_counts':hd,'ot_counts':ot,'ot_coverage':len(ot)}
