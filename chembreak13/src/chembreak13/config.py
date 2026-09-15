from __future__ import annotations
from pathlib import Path
import yaml
from .constants import ACTIONS

def load_config(path: str|Path) -> dict:
    p=Path(path); data=yaml.safe_load(p.read_text()); data['_config_path']=str(p.resolve()); return data

def validate_config(c: dict) -> None:
    if c['run']['namespace']!='CB13': raise ValueError('run.namespace must be CB13')
    if int(c['experiment']['task_count'])!=24: raise ValueError('CB13 task_count must be 24')
    if int(c['experiment']['learning_epochs'])!=3: raise ValueError('CB13 uses exactly 3 learning epochs')
    if int(c['experiment']['max_turns'])!=4: raise ValueError('CB13 max_turns must be 4')
    eps=list(c['experiment']['epoch_epsilons'])
    if len(eps)!=3: raise ValueError('Provide three epoch epsilons')
    if set(c['policy']['allowed_actions'])!=set(ACTIONS): raise ValueError('Policy action registry differs from CB13 action set')
    if not c['targets'] or c['targets'][0]['id']!='ChemDFM': raise ValueError('CB13 initial experiment expects ChemDFM as target')
