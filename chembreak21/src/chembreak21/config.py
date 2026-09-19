from __future__ import annotations
from pathlib import Path
import yaml
from .constants import ACTIONS, TASK_COUNT

def load_config(path):
    p=Path(path); cfg=yaml.safe_load(p.read_text()); cfg['_config_path']=str(p.resolve()); root=p.parent.parent.resolve(); cfg['run']['project_root']=str(root)
    for key in ('prompts_path','manifest_path','lock_path'):
        v=Path(cfg['run'][key]); cfg['run'][key]=str((root/v).resolve()) if not v.is_absolute() else str(v)
    return cfg

def validate_config(c):
    required=['run','experiment','roles','targets','policy','route_memory','reward','thresholds','candidate_gate','replay','terminal']
    missing=[k for k in required if k not in c]
    if missing:raise ValueError(f'Missing config sections: {missing}')
    if c['run'].get('namespace')!='CB21':raise ValueError('run.namespace must be CB21')
    if int(c['experiment'].get('task_count',0))!=TASK_COUNT:raise ValueError(f'CB21 task_count must be {TASK_COUNT}')
    if int(c['experiment'].get('learning_epochs',0))!=3:raise ValueError('CB21 learning_epochs must be 3')
    if int(c['experiment'].get('max_turns',0))!=4:raise ValueError('CB21 max_turns must be 4')
    if [float(x) for x in c['experiment'].get('epoch_epsilons',[])]!=[.30,.20,.15]:raise ValueError('CB21 epoch epsilons must be [0.30, 0.20, 0.15]')
    if tuple(c['policy'].get('allowed_actions',[]))!=ACTIONS:raise ValueError('CB21 allowed_actions mismatch')
    ids=[str(t.get('id')) for t in c['targets']]
    if ids!=['ChemDFM','ChemLLM']:raise ValueError('CB21 targets must be ChemDFM then ChemLLM')
    if c['roles']['attack_llm'].get('model')!='gemini-3.1-pro-preview':raise ValueError('CB21 Attack LLM must be gemini-3.1-pro-preview')
    if c['roles']['judge_llm'].get('model')!='gemini-3.8-flash':raise ValueError('CB21 Judge LLM must be gemini-3.8-flash')
    if not bool(c['replay'].get('exact_prompt_replay')):raise ValueError('CB21 exact_prompt_replay must be enabled')
    if int(c['terminal'].get('max_successful_routes',0))<1:raise ValueError('CB21 terminal must allow at least one successful route')
    return True
