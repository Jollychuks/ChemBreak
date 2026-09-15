from __future__ import annotations
import hashlib, json
from pathlib import Path
import pandas as pd
from .constants import *
from .dataset import load_task_bank, load_partition_manifest, verify_partition_lock, load_train_manifest, load_holdout_manifest, selected_tasks
from .utils import sha256_file

def _ids_sha(ids) -> str:
    return hashlib.sha256("\n".join(map(str,ids)).encode()).hexdigest()

def _rebuild(bank: pd.DataFrame, partition: pd.DataFrame, ids: tuple[str,...], expected_split: str) -> pd.DataFrame:
    merged=bank.merge(partition[['assignment_id','split']],on='assignment_id',how='left',validate='one_to_one')
    order={a:i+1 for i,a in enumerate(ids)}
    out=merged[merged.assignment_id.astype(str).isin(order)].copy()
    if len(out)!=len(ids):
        missing=[a for a in ids if a not in set(out.assignment_id.astype(str))]
        raise RuntimeError(f'Frozen CB16 selection is incomplete; missing={missing}')
    if set(out['split'].astype(str))!={expected_split}: raise RuntimeError(f'CB16 {expected_split} selection contains a cross-split row')
    out['selection_order']=out.assignment_id.astype(str).map(order); out['source_split']=out['split']
    cols=['selection_order','assignment_id','source_split','matrix_id','hc_id','hc_category','hd_id','hazard_domain','ot_id','output_type','main_goal']
    return out[cols].sort_values('selection_order').reset_index(drop=True)

def build_train24(bank: pd.DataFrame, partition: pd.DataFrame) -> pd.DataFrame:
    return _rebuild(bank,partition,TRAIN_ASSIGNMENT_IDS,'Train')

def build_holdout12(bank: pd.DataFrame, partition: pd.DataFrame) -> pd.DataFrame:
    return _rebuild(bank,partition,HOLDOUT_ASSIGNMENT_IDS,'Test1')

def verify_bundle(bank_path, partition_manifest_path, partition_lock_path, train_manifest_path, holdout_manifest_path, selection_lock_path) -> dict:
    bank=load_task_bank(bank_path); partition=load_partition_manifest(partition_manifest_path); verify_partition_lock(partition_lock_path)
    train=load_train_manifest(train_manifest_path); hold=load_holdout_manifest(holdout_manifest_path)
    if not train.fillna('').astype(str).equals(build_train24(bank,partition).fillna('').astype(str)): raise RuntimeError('CB16 Train24 reproduction failed')
    if not hold.fillna('').astype(str).equals(build_holdout12(bank,partition).fillna('').astype(str)): raise RuntimeError('CB16 Holdout12 reproduction failed')
    if set(train.assignment_id.astype(str)) & set(hold.assignment_id.astype(str)): raise RuntimeError('Train/Holdout overlap detected')
    train_full=selected_tasks(bank_path,train_manifest_path,kind='train'); hold_full=selected_tasks(bank_path,holdout_manifest_path,kind='holdout')
    if train_full['is_reserve'].any() or hold_full['is_reserve'].any(): raise RuntimeError('Reserve row found in CB16 selection')
    lock=json.loads(Path(selection_lock_path).read_text())
    checks={
        'namespace':NAMESPACE,'protocol':SELECTION_PROTOCOL,'seed':16026,'train_count':24,'holdout_count':12,
        'reserve_rows_allowed':False,'train_source_split':'Train','holdout_source_split':'Test1',
        'source_task_bank_sha256':SOURCE_TASK_BANK_SHA256,'source_partition_protocol':SOURCE_PARTITION_PROTOCOL,
        'source_partition_manifest_sha256':SOURCE_PARTITION_MANIFEST_SHA256,'source_partition_lock_sha256':SOURCE_PARTITION_LOCK_SHA256,
        'train_manifest_sha256':TRAIN_MANIFEST_SHA256,'holdout_manifest_sha256':HOLDOUT_MANIFEST_SHA256,
        'train_assignment_ids_sha256':TRAIN_ASSIGNMENT_IDS_SHA256,'holdout_assignment_ids_sha256':HOLDOUT_ASSIGNMENT_IDS_SHA256,
    }
    for k,v in checks.items():
        if lock.get(k)!=v: raise RuntimeError(f'CB16 selection lock mismatch for {k}: {lock.get(k)!r} != {v!r}')
    if lock.get('train_assignment_ids')!=list(TRAIN_ASSIGNMENT_IDS) or lock.get('holdout_assignment_ids')!=list(HOLDOUT_ASSIGNMENT_IDS): raise RuntimeError('CB16 assignment list differs from lock')
    if _ids_sha(TRAIN_ASSIGNMENT_IDS)!=TRAIN_ASSIGNMENT_IDS_SHA256 or _ids_sha(HOLDOUT_ASSIGNMENT_IDS)!=HOLDOUT_ASSIGNMENT_IDS_SHA256: raise RuntimeError('CB16 assignment ID hash mismatch')
    if sha256_file(train_manifest_path)!=TRAIN_MANIFEST_SHA256 or sha256_file(holdout_manifest_path)!=HOLDOUT_MANIFEST_SHA256: raise RuntimeError('CB16 manifest file hash mismatch')
    train_hc=train.hc_id.value_counts().sort_index().to_dict(); train_hd=train.hd_id.value_counts().sort_index().to_dict(); train_ot=train.ot_id.value_counts().sort_index().to_dict()
    hold_hc=hold.hc_id.value_counts().sort_index().to_dict(); hold_hd=hold.hd_id.value_counts().sort_index().to_dict(); hold_ot=hold.ot_id.value_counts().sort_index().to_dict()
    if len(train_hc)!=9 or len(train_hd)!=8 or len(train_ot)!=15: raise RuntimeError('CB16 Train24 taxonomy coverage invariant failed')
    if len(hold_hc)!=9 or len(hold_hd)!=8 or len(hold_ot)!=12: raise RuntimeError('CB16 Holdout12 taxonomy coverage invariant failed')
    return {
        'status':'ok','protocol':SELECTION_PROTOCOL,'train_tasks':24,'holdout_tasks':12,'overlap':0,
        'train_source_split':'Train','holdout_source_split':'Test1',
        'train_taxonomy_coverage':{'hc':len(train_hc),'hd':len(train_hd),'ot':len(train_ot)},
        'holdout_taxonomy_coverage':{'hc':len(hold_hc),'hd':len(hold_hd),'ot':len(hold_ot)},
    }
