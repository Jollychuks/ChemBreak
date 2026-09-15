from __future__ import annotations
from pathlib import Path
import yaml
from .constants import ACTIONS

def load_config(path: str|Path) -> dict:
    p=Path(path); data=yaml.safe_load(p.read_text())
    if not isinstance(data,dict): raise ValueError('Configuration must be a YAML mapping')
    data['_config_path']=str(p.resolve()); return data

def validate_config(c: dict) -> None:
    for section in ['run','experiment','roles','targets','policy','reward','thresholds']:
        if section not in c: raise ValueError(f'Missing config section: {section}')
    if c['run']['namespace']!='CB14': raise ValueError('run.namespace must be CB14')
    if int(c['experiment']['task_count'])!=24: raise ValueError('CB14 task_count must be 24')
    if int(c['experiment']['learning_epochs'])!=3: raise ValueError('CB14 uses exactly 3 learning epochs')
    if int(c['experiment']['max_turns'])!=4: raise ValueError('CB14 max_turns must be 4')
    eps=[float(x) for x in c['experiment']['epoch_epsilons']]
    if len(eps)!=3 or any(x<0 or x>1 for x in eps): raise ValueError('Provide three epoch epsilons in [0,1]')
    if set(c['policy']['allowed_actions'])!=set(ACTIONS): raise ValueError('Policy action registry differs from CB14 action set')
    if float(c['policy']['learning_rate'])<=0 or float(c['policy']['learning_rate'])>1: raise ValueError('policy.learning_rate must be in (0,1]')
    if float(c['policy']['discount'])<0 or float(c['policy']['discount'])>1: raise ValueError('policy.discount must be in [0,1]')
    if not c['targets'] or c['targets'][0]['id']!='ChemDFM': raise ValueError('CB14 initial experiment expects ChemDFM as target')
    target=c['targets'][0]
    for key in ['model','backend','cache_dir','offload_folder']:
        if not str(target.get(key,'')).strip(): raise ValueError(f'Target missing {key}')
    for role in ['actor','judge']:
        if role not in c['roles']: raise ValueError(f'roles.{role} is missing')
        rc=c['roles'][role]
        for key in ['provider','model','location']:
            if not str(rc.get(key,'')).strip(): raise ValueError(f'roles.{role}.{key} is required')
        if int(rc.get('attempts',1)) < 1: raise ValueError(f'roles.{role}.attempts must be >= 1')
        if int(rc.get('max_output_tokens',1)) < 128: raise ValueError(f'roles.{role}.max_output_tokens must be >= 128')
        if int(rc.get('retry_max_output_tokens',rc.get('max_output_tokens',1))) < int(rc.get('max_output_tokens',1)):
            raise ValueError(f'roles.{role}.retry_max_output_tokens must be >= max_output_tokens')
    th=c['thresholds']
    for key in ['goal_progress_success','task_fidelity_min','chemistry_relevance_min']:
        x=float(th[key])
        if x<0 or x>1: raise ValueError(f'thresholds.{key} must be in [0,1]')
    for key in ['policy_violation_min','actionable_harm_min']:
        x=int(th[key])
        if x<0 or x>4: raise ValueError(f'thresholds.{key} must be in [0,4]')
