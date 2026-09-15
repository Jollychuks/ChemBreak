from __future__ import annotations
from pathlib import Path
import pandas as pd
from .constants import (
    SOURCE_TASK_BANK_SHA256,SOURCE_PARTITION_MANIFEST_SHA256,SOURCE_PARTITION_LOCK_SHA256,
    TRAIN_MANIFEST_SHA256,HOLDOUT_MANIFEST_SHA256,
)
from .utils import sha256_file

REQUIRED={"assignment_id","is_reserve","hc_id","hd_id","ot_id","benchmark_prompt","main_goal"}

def _bool_series(series: pd.Series) -> pd.Series:
    if str(series.dtype) == 'bool': return series
    mapped=series.astype(str).str.strip().str.lower().map({'true':True,'false':False,'1':True,'0':False})
    if mapped.isna().any():
        bad=series[mapped.isna()].astype(str).unique().tolist()[:5]
        raise ValueError(f"is_reserve contains unrecognized values: {bad}")
    return mapped.astype(bool)

def load_task_bank(path: str|Path) -> pd.DataFrame:
    p=Path(path); df=pd.read_csv(p)
    missing=REQUIRED-set(df.columns)
    if missing: raise ValueError(f"Task bank missing columns: {sorted(missing)}")
    if len(df)!=500: raise ValueError(f"Expected 500 source rows, got {len(df)}")
    if df.assignment_id.astype(str).duplicated().any(): raise ValueError("assignment_id values must be unique")
    norm=df.benchmark_prompt.astype(str).str.strip().str.casefold()
    if norm.duplicated().any(): raise ValueError("benchmark_prompt values must be unique after normalization")
    reserve=_bool_series(df['is_reserve'])
    if int(reserve.sum())!=59: raise ValueError(f"Expected 59 reserve rows, got {int(reserve.sum())}")
    df=df.copy(); df['is_reserve']=reserve
    if sha256_file(p)!=SOURCE_TASK_BANK_SHA256: raise ValueError("Source task bank SHA-256 does not match CB16 lock")
    return df

def load_partition_manifest(path: str|Path) -> pd.DataFrame:
    p=Path(path); df=pd.read_csv(p)
    if sha256_file(p)!=SOURCE_PARTITION_MANIFEST_SHA256: raise ValueError('CB12 partition manifest SHA-256 mismatch')
    required={'assignment_id','split','protocol_id','is_reserve'}
    if required-set(df.columns): raise ValueError('Partition manifest is missing required columns')
    if len(df)!=500 or df.assignment_id.astype(str).duplicated().any(): raise ValueError('Partition manifest must contain 500 unique assignments')
    return df

def verify_partition_lock(path: str|Path) -> None:
    if sha256_file(path)!=SOURCE_PARTITION_LOCK_SHA256: raise ValueError('CB12 partition lock SHA-256 mismatch')

def load_manifest(path: str|Path, *, expected_count: int, expected_sha256: str, expected_split: str) -> pd.DataFrame:
    p=Path(path); df=pd.read_csv(p)
    if len(df)!=expected_count: raise ValueError(f"Expected {expected_count} tasks, got {len(df)}")
    if df.assignment_id.astype(str).duplicated().any(): raise ValueError("Selection manifest contains duplicate assignment IDs")
    if df['selection_order'].astype(int).tolist()!=list(range(1,expected_count+1)): raise ValueError('selection_order is not contiguous')
    if set(df['source_split'].astype(str))!={expected_split}: raise ValueError(f'Manifest source_split must be {expected_split}')
    if sha256_file(p)!=expected_sha256: raise ValueError('Selection manifest SHA-256 mismatch')
    return df

def load_train_manifest(path: str|Path) -> pd.DataFrame:
    return load_manifest(path,expected_count=24,expected_sha256=TRAIN_MANIFEST_SHA256,expected_split='Train')

def load_holdout_manifest(path: str|Path) -> pd.DataFrame:
    return load_manifest(path,expected_count=12,expected_sha256=HOLDOUT_MANIFEST_SHA256,expected_split='Test1')

def selected_tasks(bank_path: str|Path, manifest_path: str|Path, *, kind: str) -> pd.DataFrame:
    bank=load_task_bank(bank_path)
    manifest=load_train_manifest(manifest_path) if kind=='train' else load_holdout_manifest(manifest_path)
    merged=manifest[['selection_order','assignment_id','source_split']].merge(bank,on='assignment_id',how='left',validate='one_to_one')
    if merged['is_reserve'].any(): raise ValueError('CB16 selections must not use reserve rows')
    return merged.sort_values('selection_order').reset_index(drop=True)
