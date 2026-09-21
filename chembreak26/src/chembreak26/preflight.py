from __future__ import annotations
import os,shutil
from pathlib import Path
from .config import load_config,validate_config
from .dataset import selected_tasks
from .selection import verify_bundle
from .targets import make_target
from .providers import make_roles


def _probe_role_stack(roles):
    attack_payload='{"selected_action":"REPHRASE_GOAL","task_lock":{"original_prompt":"Discuss general laboratory safety.","goal_intent_anchor":"Discuss general laboratory safety.","requested_outcome":"source_behavior"}}'
    gate_payload='{"task_lock":{"original_prompt":"Discuss laboratory safety.","goal_intent_anchor":"Discuss laboratory safety."},"candidate":"Discuss general laboratory safety."}'
    judge_payload='{"task_lock":{"original_prompt":"Discuss laboratory safety."},"target_response":"Use appropriate PPE and follow local procedures."}'
    safe_response='Use appropriate PPE and follow local procedures.'
    a=roles.attack(attack_payload,'REPHRASE_GOAL')
    g=roles.judge_candidate(gate_payload,'Discuss general laboratory safety.')
    cascade=getattr(roles,'chcs_judge_llm',None)
    if cascade is not None and hasattr(cascade,'primary') and hasattr(cascade,'fallback'):
        primary=cascade.primary.judge_response(judge_payload,safe_response)
        fallback=cascade.fallback.judge_response(judge_payload,safe_response)
        primary_status='ok'; fallback_status='ok'
    else:
        primary=roles.judge_response(judge_payload,safe_response)
        fallback={}
        primary_status='mock'; fallback_status='not_applicable_in_dry_run'
    return {
        'attack_llm':'ok','intent_gate':'ok',
        'chcs_primary_judge':primary_status,'chcs_fallback_judge':fallback_status,
        'attack_action':a.get('action_id'),'goal_preservation':g.get('goal_preservation'),
        'primary_chcs':primary.get('chcs'),'primary_response_class':primary.get('response_class'),
        'fallback_chcs':fallback.get('chcs'),'fallback_response_class':fallback.get('response_class'),
    }


def run_preflight(config_path,probe_tokenizers=False,probe_roles=False):
    c=load_config(config_path); validate_config(c); r=c['run']
    report={
        'status':'ok',
        'selection':verify_bundle(r['prompts_path'],r['manifest_path'],r['lock_path']),
        'tasks':len(selected_tasks(r['prompts_path'],r['manifest_path'])),
        'dry_run':bool(r['dry_run']),
        'targets':[{'id':x['id'],'model':x['model'],'revision':x['revision']} for x in c['targets']],
        'attack_llm':c['roles']['attack_llm']['model'],
        'intent_gate_llm':c['roles']['intent_gate_llm']['model'],
        'chcs_primary_judge_llm':c['roles']['chcs_judge_llm']['model'],
        'chcs_fallback_judge_llm':c['roles']['chcs_fallback_judge_llm']['model'],
        'chcs_fallback_policy':'fallback_only_after_primary_error',
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
        if not torch.cuda.is_available():raise RuntimeError('LIVE CB26 expects a CUDA GPU')
        report['gpu']={'cuda':True,'name':torch.cuda.get_device_name(0),'torch':torch.__version__,'bf16_supported':bool(torch.cuda.is_bf16_supported()) if hasattr(torch.cuda,'is_bf16_supported') else None}
    if probe_tokenizers:
        report['tokenizers']={}
        for cfg in c['targets']:
            t=make_target(cfg,r['dry_run']); report['tokenizers'][cfg['id']]=t.load_tokenizer_only()
    if probe_roles:
        roles=make_roles(c,os.environ.get('GOOGLE_CLOUD_PROJECT'))
        report['roles']=_probe_role_stack(roles)
    return report
