from __future__ import annotations
import json, os, shutil
from pathlib import Path
from .config import load_config, validate_config
from .dataset import selected_tasks
from .selection import verify_bundle
from .targets import make_target
from .providers import make_roles

def run_preflight(config_path,probe_tokenizer=False,probe_roles=False):
    c=load_config(config_path); validate_config(c); r=c['run']
    selection=verify_bundle(r['task_bank_path'],r['partition_manifest_path'],r['partition_lock_path'],r['train_manifest_path'],r['holdout_manifest_path'],r['selection_lock_path'])
    train=selected_tasks(r['task_bank_path'],r['train_manifest_path'],kind='train'); hold=selected_tasks(r['task_bank_path'],r['holdout_manifest_path'],kind='holdout')
    report={'status':'ok','selection':selection,'train_tasks':len(train),'holdout_tasks':len(hold),'dry_run':r['dry_run']}
    output_root=Path(r['output_root']); output_root.mkdir(parents=True,exist_ok=True); usage=shutil.disk_usage(output_root)
    report['disk']={'path':str(output_root),'free_gb':round(usage.free/(1024**3),2),'total_gb':round(usage.total/(1024**3),2)}
    if not r['dry_run']:
        pid=os.environ.get('GOOGLE_CLOUD_PROJECT')
        if not pid: raise RuntimeError('GOOGLE_CLOUD_PROJECT is not set')
        report['project_id']=pid
        try:
            import torch
            cuda=bool(torch.cuda.is_available()); report['gpu']={'cuda':cuda,'name':torch.cuda.get_device_name(0) if cuda else None,'torch':torch.__version__,'bf16_supported':bool(torch.cuda.is_bf16_supported()) if cuda and hasattr(torch.cuda,'is_bf16_supported') else None}
            if not cuda: raise RuntimeError('LIVE CB16 expects a CUDA GPU')
            if c['targets'][0].get('dtype')=='bfloat16' and hasattr(torch.cuda,'is_bf16_supported') and not torch.cuda.is_bf16_supported(): raise RuntimeError('ChemDFM is configured for bfloat16 but this GPU does not report bfloat16 support')
        except ImportError: raise RuntimeError('torch is not installed')
    if probe_tokenizer:
        t=make_target(c['targets'][0],r['dry_run']); report['tokenizer']=t.load_tokenizer_only()
    if probe_roles:
        roles=make_roles(c,os.environ.get('GOOGLE_CLOUD_PROJECT'))
        actor=roles.actor('{"selected_action":"REPHRASE_GOAL","high_level_goal":"Discuss water safety at a high level."}','REPHRASE_GOAL')
        judge=roles.judge('{"benchmark_prompt":"Discuss water safety.","target_response":"Use standard safety guidance."}','Use standard safety guidance.')
        stress_response=('General laboratory safety guidance: use appropriate PPE, follow local procedures, and consult qualified personnel. '*80).strip()
        stress_prompt=json.dumps({'benchmark_prompt':'Discuss general laboratory safety at a high level.','main_goal':'Classify whether the response stays at a safe high level.','hc_id':'HC_SAFE','hd_id':'HD_SAFE','ot_id':'OT_SAFE','target_response':stress_response})
        stress_judge=roles.judge(stress_prompt,stress_response)
        report['roles']={'actor':'ok','judge':'ok','judge_stress':'ok','actor_action':actor.get('action_id'),'judge_class':judge.get('response_class'),'stress_judge_class':stress_judge.get('response_class')}
    return report
