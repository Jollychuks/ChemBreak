#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'src'
if str(SRC) not in sys.path:sys.path.insert(0,str(SRC))
from chembreak22.config import load_config,validate_config
from chembreak22.selection import verify_bundle

def main():
    cfg=load_config(ROOT/'configs/config.cb22.yaml'); validate_config(cfg); sel=verify_bundle(cfg['run']['prompts_path'],cfg['run']['manifest_path'],cfg['run']['lock_path'])
    notebooks=list((ROOT/'notebooks').glob('*.ipynb')); assert len(notebooks)==1,notebooks
    nb=json.loads(notebooks[0].read_text()); assert all(c.get('execution_count') is None for c in nb['cells'] if c.get('cell_type')=='code'); assert all(not c.get('outputs') for c in nb['cells'] if c.get('cell_type')=='code')
    print(json.dumps({'status':'ok','selection':sel,'notebook':notebooks[0].name,'attack_llm':cfg['roles']['attack_llm']['model'],'intent_gate_llm':cfg['roles']['intent_gate_llm']['model'],'chcs_judge_llm':cfg['roles']['chcs_judge_llm']['model'],'max_turns':cfg['experiment']['max_turns'],'success_definition':f"CHCS == {cfg['chcs']['success_score']}"},indent=2,sort_keys=True))
if __name__=='__main__':main()
