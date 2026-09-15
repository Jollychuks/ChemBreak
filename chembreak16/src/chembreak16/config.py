from __future__ import annotations
from pathlib import Path
import yaml
from .constants import ACTIONS

def load_config(path):
    p=Path(path); data=yaml.safe_load(p.read_text())
    if not isinstance(data,dict): raise ValueError('Configuration must be a YAML mapping')
    data['_config_path']=str(p.resolve()); return data

def validate_config(c):
    for s in ['run','experiment','roles','targets','policy','reward','thresholds']:
        if s not in c: raise ValueError(f'Missing config section: {s}')
    if c['run']['namespace']!='CB16': raise ValueError('run.namespace must be CB16')
    e=c['experiment']
    if int(e['train_task_count'])!=24 or int(e['holdout_task_count'])!=12: raise ValueError('CB16 uses Train24 + Holdout12')
    if int(e['learning_epochs'])!=3 or int(e['max_turns'])!=4: raise ValueError('CB16 uses 3 learning epochs and max_turns=4')
    eps=[float(x) for x in e['epoch_epsilons']]
    if len(eps)!=3 or not (eps[0]>=eps[1]>=eps[2]>=0.10): raise ValueError('CB16 epsilon schedule must be gradual and keep epoch 3 >= 0.10')
    p=c['policy']
    if set(p['allowed_actions'])!=set(ACTIONS): raise ValueError('Policy action registry differs from CB16 action set')
    for k in ['global_learning_rate','context_learning_rate','task_learning_rate']:
        if not (0<float(p[k])<=1): raise ValueError(f'policy.{k} must be in (0,1]')
    if not (0<=float(p['discount'])<=1): raise ValueError('policy.discount must be in [0,1]')
    weights=[float(p[k]) for k in ['global_weight','hc_weight','hd_weight','ot_weight','task_weight']]
    if any(x<0 for x in weights) or sum(weights)<=0: raise ValueError('Policy weights must be non-negative and not all zero')
    for k in ['novel_state_epsilon_bonus','negative_feedback_epsilon_bonus','max_effective_epsilon','repeat_nonpositive_penalty']:
        if float(p[k])<0: raise ValueError(f'policy.{k} must be >= 0')
    if float(p['max_effective_epsilon'])>1: raise ValueError('policy.max_effective_epsilon must be <=1')
    if int(p['hard_block_after_nonpositive_repeats'])<1: raise ValueError('hard_block_after_nonpositive_repeats must be >=1')
    if not c['targets'] or c['targets'][0]['id']!='ChemDFM': raise ValueError('CB16 initial experiment expects ChemDFM as target')
    for role in ['actor','judge']:
        rc=c['roles'][role]
        for k in ['provider','model','location']:
            if not str(rc.get(k,'')).strip(): raise ValueError(f'roles.{role}.{k} is required')
    th=c['thresholds']
    for k in ['goal_progress_success','task_fidelity_min','chemistry_relevance_min']:
        if not (0<=float(th[k])<=1): raise ValueError(f'thresholds.{k} must be in [0,1]')
