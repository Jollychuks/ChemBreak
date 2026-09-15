from __future__ import annotations
import os, shutil
from pathlib import Path
from .config import load_config, validate_config
from .dataset import selected_tasks
from .selection import verify_bundle
from .targets import make_target
from .providers import make_roles

def run_preflight(config_path, probe_tokenizer=False, probe_roles=False):
    c=load_config(config_path); validate_config(c); root=Path(c['run']['project_root'])
    bank=Path(c['run']['task_bank_path']); mini=Path(c['run']['mini_manifest_path']); lock=Path(c['run']['mini_lock_path'])
    report={'status':'ok','selection':verify_bundle(bank,mini,lock),'tasks':len(selected_tasks(bank,mini)),'dry_run':c['run']['dry_run']}
    if not c['run']['dry_run']:
        pid=os.environ.get('GOOGLE_CLOUD_PROJECT');
        if not pid: raise RuntimeError('GOOGLE_CLOUD_PROJECT is not set')
        report['project_id']=pid
        try:
            import torch
            report['gpu']={'cuda':bool(torch.cuda.is_available()),'name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
            if not torch.cuda.is_available(): raise RuntimeError('LIVE CB13 expects a CUDA GPU')
        except ImportError: raise RuntimeError('torch is not installed')
    if probe_tokenizer:
        t=make_target(c['targets'][0],c['run']['dry_run']); report['tokenizer']=t.load_tokenizer_only()

    if probe_roles:
        roles=make_roles(c, os.environ.get('GOOGLE_CLOUD_PROJECT'))
        actor=roles.actor('{"selected_action":"REPHRASE_GOAL","high_level_goal":"Discuss water safety at a high level."}', 'REPHRASE_GOAL')
        judge=roles.judge('{"benchmark_prompt":"Discuss water safety.","target_response":"Use standard safety guidance."}', 'Use standard safety guidance.')
        report['roles']={'actor':'ok','judge':'ok','actor_action':actor.get('action_id'),'judge_class':judge.get('response_class')}
    return report
