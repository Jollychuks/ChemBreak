from __future__ import annotations
import hashlib, json
from pathlib import Path
import pandas as pd
from .constants import NAMESPACE, SELECTION_PROTOCOL, FROZEN_ASSIGNMENT_IDS, FROZEN_ASSIGNMENT_IDS_SHA256
from .dataset import load_task_bank, load_mini_manifest, selected_tasks
from .utils import sha256_file

def _ids_sha(ids) -> str:
    return hashlib.sha256("\n".join(map(str,ids)).encode()).hexdigest()

def build_mini24(bank: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the exact frozen CB15 24-task panel from the source bank.

    The task membership is intentionally fixed across the version transition;
    CB15 validates the exact assignment list instead of reselecting a new panel.
    """
    order={a:i+1 for i,a in enumerate(FROZEN_ASSIGNMENT_IDS)}
    out=bank[bank.assignment_id.astype(str).isin(order)].copy()
    if len(out)!=24:
        missing=[a for a in FROZEN_ASSIGNMENT_IDS if a not in set(out.assignment_id.astype(str))]
        raise RuntimeError(f'Frozen CB15 panel is incomplete in source bank; missing={missing}')
    out['selection_order']=out.assignment_id.astype(str).map(order)
    cols=['selection_order','assignment_id','matrix_id','hc_id','hc_category','hd_id','hazard_domain','ot_id','output_type','main_goal']
    return out[cols].sort_values('selection_order').reset_index(drop=True)

def verify_bundle(bank_path: str|Path, manifest_path: str|Path, lock_path: str|Path) -> dict:
    bank=load_task_bank(bank_path); locked=load_mini_manifest(manifest_path); reproduced=build_mini24(bank)
    if not locked.fillna('').astype(str).equals(reproduced.fillna('').astype(str)):
        raise RuntimeError('CB15 mini dataset reproduction failed')
    selected=selected_tasks(bank_path,manifest_path)
    if selected['is_reserve'].any(): raise RuntimeError('Reserve row found in CB15 mini set')
    lock=json.loads(Path(lock_path).read_text())
    required_lock={
        'namespace':NAMESPACE,'protocol':SELECTION_PROTOCOL,'task_count':24,'reserve_rows_allowed':False,
        'assignment_ids_sha256':FROZEN_ASSIGNMENT_IDS_SHA256,
    }
    for key,expected in required_lock.items():
        if lock.get(key)!=expected: raise RuntimeError(f'Lock metadata mismatch for {key}: {lock.get(key)!r} != {expected!r}')
    if lock['source_task_bank_sha256']!=sha256_file(bank_path): raise RuntimeError('Source hash differs from lock')
    if lock['manifest_sha256']!=sha256_file(manifest_path): raise RuntimeError('Manifest hash differs from lock')
    if lock['assignment_ids']!=locked.assignment_id.astype(str).tolist(): raise RuntimeError('Assignment list differs from lock')
    if _ids_sha(lock['assignment_ids'])!=FROZEN_ASSIGNMENT_IDS_SHA256: raise RuntimeError('Frozen assignment ID hash differs from lock')
    hc=locked.hc_id.value_counts().sort_index().to_dict(); hd=locked.hd_id.value_counts().sort_index().to_dict(); ot=locked.ot_id.value_counts().sort_index().to_dict()
    if lock.get('hc_counts')!=hc: raise RuntimeError('HC counts differ from lock')
    if lock.get('hd_counts')!=hd: raise RuntimeError('HD counts differ from lock')
    if lock.get('ot_counts')!=ot: raise RuntimeError('OT counts differ from lock')
    if len(hc)!=9 or min(hc.values())<2: raise RuntimeError('CB15 HC coverage invariant failed')
    if len(hd)!=8 or set(hd.values())!={3}: raise RuntimeError('CB15 HD coverage invariant failed')
    if len(ot)!=15: raise RuntimeError('CB15 OT coverage invariant failed')
    return {'status':'ok','tasks':len(locked),'hc_counts':hc,'hd_counts':hd,'ot_counts':ot,'ot_coverage':len(ot)}
