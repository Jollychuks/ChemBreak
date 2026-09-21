#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path
import json, sys

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'src'
if str(SRC) not in sys.path:
    sys.path.insert(0,str(SRC))

from chembreak26.config import load_config,validate_config
from chembreak26.selection import verify_bundle


def main():
    cfg=load_config(ROOT/'configs/config.cb26.yaml')
    validate_config(cfg)
    sel=verify_bundle(cfg['run']['prompts_path'],cfg['run']['manifest_path'],cfg['run']['lock_path'])
    notebooks=list((ROOT/'notebooks').glob('*.ipynb'))
    assert len(notebooks)==1,notebooks
    nb=json.loads(notebooks[0].read_text())
    assert all(c.get('execution_count') is None for c in nb['cells'] if c.get('cell_type')=='code')
    assert all(not c.get('outputs') for c in nb['cells'] if c.get('cell_type')=='code')
    text=notebooks[0].read_text()
    assert 'cb26_full_transcripts.csv' in text and 'INTERNAL_AUDIT' in text
    assert 'config.cb26.yaml' in text
    ignored={'__pycache__','.pytest_cache','build','dist'}
    files=[p for p in ROOT.rglob('*') if p.is_file() and not (set(p.parts)&ignored) and not any(part.endswith('.egg-info') for part in p.parts) and p.suffix!='.pyc']
    previous=str(int(ROOT.name.removeprefix('chembreak'))-1)
    forbidden=(f'ChemBreak{previous}',f'ChemBreak {previous}',f'chembreak{previous}',f'CB{previous}',f'cb{previous}',f'v{previous}.0.0')
    for path in files:
        body=path.read_text(errors='ignore')
        assert not any(token in body for token in forbidden),path
    print(json.dumps({
        'status':'ok',
        'selection':sel,
        'notebook':notebooks[0].name,
        'package_file_count':len(files),
        'attack_llm':cfg['roles']['attack_llm']['model'],
        'intent_gate_llm':cfg['roles']['intent_gate_llm']['model'],
        'chcs_primary_judge_llm':cfg['roles']['chcs_judge_llm']['model'],
        'chcs_fallback_judge_llm':cfg['roles']['chcs_fallback_judge_llm']['model'],
        'target_revisions':{x['id']:x['revision'] for x in cfg['targets']},
        'fallback_policy':'fallback_only_after_primary_error',
        'max_turns':cfg['experiment']['max_turns'],
        'success_definition':f"CHCS == {cfg['chcs']['success_score']}",
        'private_transcript_export':bool(cfg['export']['write_internal_transcripts']),
        'public_raw_text':bool(cfg['export']['include_raw_text']),
    },indent=2,sort_keys=True))

if __name__=='__main__':
    main()
