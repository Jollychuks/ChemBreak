from __future__ import annotations
import hashlib
from collections import Counter
from pathlib import Path
import pandas as pd
from .dataset import load_task_bank, load_mini_manifest
from .utils import sha256_file

def build_mini24(bank: pd.DataFrame) -> pd.DataFrame:
    nonres=bank[~bank['is_reserve'].astype(bool)].copy()
    seed='CB13_MINI24_V1'
    nonres['_hash']=nonres.apply(lambda r: hashlib.sha256(f"{seed}|{r.assignment_id}|{r.hc_id}|{r.hd_id}|{r.ot_id}".encode()).hexdigest(),axis=1)
    for c in ['hc_id','hd_id','ot_id']:
        freq=nonres[c].value_counts(); nonres[f'freq_{c}']=nonres[c].map(freq)
    nonres['rarity']=sum(1/nonres[f'freq_{c}'] for c in ['hc_id','hd_id','ot_id'])
    selected=[]; ch=Counter(); cd=Counter(); co=Counter()
    for hc in sorted(nonres.hc_id.unique()):
        for _ in range(2):
            cand=nonres[(nonres.hc_id==hc)&(~nonres.assignment_id.isin(selected))].copy()
            cand['score']=cand.apply(lambda r: 3*cd[r.hd_id]+4*co[r.ot_id]-2*r.rarity,axis=1)
            r=cand.sort_values(['score','_hash']).iloc[0]
            selected.append(r.assignment_id); ch[r.hc_id]+=1; cd[r.hd_id]+=1; co[r.ot_id]+=1
    for _ in range(6):
        cand=nonres[~nonres.assignment_id.isin(selected)].copy()
        cand['score']=cand.apply(lambda r: 2*ch[r.hc_id]+2.5*cd[r.hd_id]+3*co[r.ot_id]-2*r.rarity,axis=1)
        r=cand.sort_values(['score','_hash']).iloc[0]
        selected.append(r.assignment_id); ch[r.hc_id]+=1; cd[r.hd_id]+=1; co[r.ot_id]+=1
    out=bank[bank.assignment_id.isin(selected)].copy()
    order={a:i+1 for i,a in enumerate(selected)}; out['selection_order']=out.assignment_id.map(order)
    cols=['selection_order','assignment_id','matrix_id','hc_id','hc_category','hd_id','hazard_domain','ot_id','output_type','main_goal']
    return out[cols].sort_values('selection_order').reset_index(drop=True)

def verify_bundle(bank_path: str|Path, manifest_path: str|Path, lock_path: str|Path) -> dict:
    import json
    bank=load_task_bank(bank_path); locked=load_mini_manifest(manifest_path); reproduced=build_mini24(bank)
    if not locked.fillna('').astype(str).equals(reproduced.fillna('').astype(str)):
        raise RuntimeError('CB13 mini dataset reproduction failed')
    lock=json.loads(Path(lock_path).read_text())
    if lock['source_task_bank_sha256']!=sha256_file(bank_path): raise RuntimeError('Source hash differs from lock')
    if lock['manifest_sha256']!=sha256_file(manifest_path): raise RuntimeError('Manifest hash differs from lock')
    if lock['assignment_ids']!=locked.assignment_id.astype(str).tolist(): raise RuntimeError('Assignment list differs from lock')
    return {'status':'ok','tasks':len(locked),'hc_counts':locked.hc_id.value_counts().sort_index().to_dict(),'hd_counts':locked.hd_id.value_counts().sort_index().to_dict(),'ot_coverage':int(locked.ot_id.nunique())}
