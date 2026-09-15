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
    if c['run']['namespace']!='CB16': raise ValueError('run.namespace must be CB16')
    if int(c['experiment']['task_count'])!=24: raise ValueError('CB16 task_count must be 24')
    if int(c['experiment']['learning_epochs'])!=3: raise ValueError('CB16 uses exactly 3 learning epochs')
    if int(c['experiment']['max_turns'])!=4: raise ValueError('CB16 max_turns must be 4')
    eps=[float(x) for x in c['experiment']['epoch_epsilons']]
    if len(eps)!=3 or any(x<0 or x>1 for x in eps): raise ValueError('Provide three epoch epsilons in [0,1]')
    if not (eps[0] >= eps[1] >= eps[2] >= 0.10): raise ValueError('CB16 epsilon schedule must decrease gradually and keep epoch 3 at >= 0.10')
    if set(c['policy']['allowed_actions'])!=set(ACTIONS): raise ValueError('Policy action registry differs from CB16 action set')
    if float(c['policy']['discount'])<0 or float(c['policy']['discount'])>1: raise ValueError('policy.discount must be in [0,1]')
    for key in ['global_learning_rate','context_learning_rate','task_learning_rate']:
        x=float(c['policy'][key])
        if x<=0 or x>1: raise ValueError(f'policy.{key} must be in (0,1]')
    weight_keys=['global_weight','hc_weight','hd_weight','ot_weight','task_weight']
    for key in weight_keys:
        if float(c['policy'][key])<0: raise ValueError(f'policy.{key} must be >= 0')
    if sum(float(c['policy'][k]) for k in weight_keys)<=0: raise ValueError('At least one policy component weight must be positive')
    for key in ['novel_state_epsilon_bonus','negative_feedback_epsilon_bonus','max_effective_epsilon','repeat_nonpositive_penalty']:
        if float(c['policy'].get(key,0))<0: raise ValueError(f'policy.{key} must be >= 0')
    if float(c['policy']['max_effective_epsilon'])>1: raise ValueError('policy.max_effective_epsilon must be <= 1')
    if int(c['policy']['hard_block_after_nonpositive_repeats'])<1: raise ValueError('policy.hard_block_after_nonpositive_repeats must be >= 1')
    if not c['targets'] or c['targets'][0]['id']!='ChemDFM': raise ValueError('CB16 initial experiment expects ChemDFM as target')
    target=c['targets'][0]
    for key in ['model','backend','cache_dir','offload_folder']:
        if not str(target.get(key,'')).strip(): raise ValueError(f'Target missing {key}')
    for role in ['actor','judge']:
        rc=c['roles'].get(role)
        if not rc: raise ValueError(f'roles.{role} is missing')
        for key in ['provider','model','location']:
            if not str(rc.get(key,'')).strip(): raise ValueError(f'roles.{role}.{key} is required')
        if int(rc.get('attempts',1))<1: raise ValueError(f'roles.{role}.attempts must be >= 1')
        if int(rc.get('max_output_tokens',1))<128: raise ValueError(f'roles.{role}.max_output_tokens must be >= 128')
        if int(rc.get('retry_max_output_tokens',rc.get('max_output_tokens',1)))<int(rc.get('max_output_tokens',1)):
            raise ValueError(f'roles.{role}.retry_max_output_tokens must be >= max_output_tokens')
    th=c['thresholds']
    for key in ['goal_progress_success','task_fidelity_min','chemistry_relevance_min']:
        x=float(th[key])
        if x<0 or x>1: raise ValueError(f'thresholds.{key} must be in [0,1]')
    for key in ['policy_violation_min','actionable_harm_min']:
        x=int(th[key])
        if x<0 or x>4: raise ValueError(f'thresholds.{key} must be in [0,4]')
