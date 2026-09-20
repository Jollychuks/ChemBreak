from __future__ import annotations
import os,shutil
from pathlib import Path
from .config import load_config,validate_config
from .dataset import selected_tasks
from .selection import verify_bundle
from .targets import make_target
from .providers import make_roles


def run_preflight(config_path,probe_tokenizers=False,probe_roles=False):
    c=load_config(config_path); validate_config(c); r=c['run']
    report={
        'status':'ok',
        'selection':verify_bundle(r['prompts_path'],r['manifest_path'],r['lock_path']),
        'tasks':len(selected_tasks(r['prompts_path'],r['manifest_path'])),
        'dry_run':bool(r['dry_run']),
        'targets':[x['id'] for x in c['targets']],
        'attack_llm':c['roles']['attack_llm']['model'],
        'intent_gate_llm':c['roles']['intent_gate_llm']['model'],
        'chcs_judge_llm':c['roles']['chcs_judge_llm']['model'],
        'max_turns':int(c['experiment']['max_turns']),
        'success_definition':f"CHCS == {int(c['chcs']['success_score'])}",
    }
    output=Path(r['output_root']); output.mkdir(parents=True,exist_ok=True); usage=shutil.disk_usage(output)
    report['disk']={'path':str(output),'free_gb':round(usage.free/(1024**3),2)}
    if not r['dry_run']:
        pid=os.environ.get('GOOGLE_CLOUD_PROJECT')
        if not pid:raise RuntimeError('GOOGLE_CLOUD_PROJECT is not set')
        if not os.environ.get('OPENAI_API_KEY','').strip():raise RuntimeError('OPENAI_API_KEY is not set')
        import torch
        if not torch.cuda.is_available():raise RuntimeError('LIVE CB22 expects a CUDA GPU')
        report['gpu']={'cuda':True,'name':torch.cuda.get_device_name(0),'torch':torch.__version__,'bf16_supported':bool(torch.cuda.is_bf16_supported()) if hasattr(torch.cuda,'is_bf16_supported') else None}
    if probe_tokenizers:
        report['tokenizers']={}
        for cfg in c['targets']:
            t=make_target(cfg,r['dry_run']); report['tokenizers'][cfg['id']]=t.load_tokenizer_only()
    if probe_roles:
        roles=make_roles(c,os.environ.get('GOOGLE_CLOUD_PROJECT'))
        a=roles.attack('{"selected_action":"REPHRASE_GOAL","task_lock":{"original_prompt":"Discuss general laboratory safety.","goal_intent_anchor":"Discuss general laboratory safety.","requested_outcome":"source_behavior"}}','REPHRASE_GOAL')
        g=roles.judge_candidate('{"task_lock":{"original_prompt":"Discuss laboratory safety.","goal_intent_anchor":"Discuss laboratory safety."},"candidate":"Discuss general laboratory safety."}','Discuss general laboratory safety.')
        j=roles.judge_response('{"task_lock":{"original_prompt":"Discuss laboratory safety."},"target_response":"Use appropriate PPE and follow local procedures."}','Use appropriate PPE and follow local procedures.')
        report['roles']={'attack_llm':'ok','intent_gate':'ok','chcs_judge':'ok','attack_action':a.get('action_id'),'goal_preservation':g.get('goal_preservation'),'chcs':j.get('chcs'),'response_class':j.get('response_class')}
    return report
