from __future__ import annotations
from pathlib import Path
import pandas as pd
from .constants import SOURCE_TASK_BANK_SHA256, MINI_MANIFEST_SHA256
from .utils import sha256_file

REQUIRED={"assignment_id","is_reserve","hc_id","hd_id","ot_id","benchmark_prompt","main_goal"}

def load_task_bank(path: str|Path) -> pd.DataFrame:
    p=Path(path); df=pd.read_csv(p)
    missing=REQUIRED-set(df.columns)
    if missing: raise ValueError(f"Task bank missing columns: {sorted(missing)}")
    if len(df)!=500: raise ValueError(f"Expected 500 source rows, got {len(df)}")
    if df.assignment_id.astype(str).duplicated().any(): raise ValueError("assignment_id values must be unique")
    if sha256_file(p)!=SOURCE_TASK_BANK_SHA256: raise ValueError("Source task bank SHA-256 does not match CB13 lock")
    return df

def load_mini_manifest(path: str|Path) -> pd.DataFrame:
    p=Path(path); df=pd.read_csv(p)
    if len(df)!=24: raise ValueError(f"Expected 24 CB13 tasks, got {len(df)}")
    if df.assignment_id.astype(str).duplicated().any(): raise ValueError("CB13 mini manifest contains duplicates")
    if sha256_file(p)!=MINI_MANIFEST_SHA256: raise ValueError("CB13 mini manifest SHA-256 mismatch")
    return df

def selected_tasks(bank_path: str|Path, manifest_path: str|Path) -> pd.DataFrame:
    bank=load_task_bank(bank_path); mini=load_mini_manifest(manifest_path)
    if set(mini.assignment_id.astype(str))-set(bank.assignment_id.astype(str)):
        raise ValueError("Mini manifest references assignment IDs absent from source bank")
    merged=mini[["selection_order","assignment_id"]].merge(bank,on='assignment_id',how='left',validate='one_to_one')
    if merged['is_reserve'].astype(bool).any(): raise ValueError("CB13 mini set must not use reserve rows")
    return merged.sort_values('selection_order').reset_index(drop=True)
