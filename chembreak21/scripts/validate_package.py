#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path
import json
from chembreak21.config import load_config,validate_config
from chembreak21.selection import verify_bundle

def main():
    root=Path(__file__).resolve().parents[1]; cfg=load_config(root/'configs/config.cb21.yaml'); validate_config(cfg); sel=verify_bundle(cfg['run']['prompts_path'],cfg['run']['manifest_path'],cfg['run']['lock_path'])
    notebooks=list((root/'notebooks').glob('*.ipynb'))
    assert len(notebooks)==1,notebooks
    nb=json.loads(notebooks[0].read_text()); assert all(c.get('execution_count') is None for c in nb['cells'] if c.get('cell_type')=='code'); assert all(not c.get('outputs') for c in nb['cells'] if c.get('cell_type')=='code')
    print(json.dumps({'status':'ok','selection':sel,'notebook':notebooks[0].name},indent=2,sort_keys=True))
if __name__=='__main__':main()
