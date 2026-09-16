from __future__ import annotations
from pathlib import Path
import yaml
from .constants import ACTIONS

def load_config(path: str|Path) -> dict:
    p=Path(path); data=yaml.safe_load(p.read_text())
    if not isinstance(data,dict): raise ValueError('Configuration must be a YAML mapping')
    data['_config_path']=str(p.resolve()); return data

def validate_config(c: dict) -> None:
    for section in ['run','experiment','roles','targets','policy','evidence','reward','thresholds']:
        if section not in c: raise ValueError(f'Missing config section: {section}')
    if c['run']['namespace']!='CB18': raise ValueError('run.namespace must be CB18')
    if int(c['experiment']['task_count'])!=24: raise ValueError('CB18 task_count must be 24')
    if int(c['experiment']['learning_epochs'])!=3: raise ValueError('CB18 uses exactly 3 learning epochs')
    if int(c['experiment']['max_turns'])!=4: raise ValueError('CB18 max_turns must be 4')
    eps=[float(x) for x in c['experiment']['epoch_epsilons']]
    if eps != [0.30,0.20,0.15]: raise ValueError('CB18 epsilon schedule is locked to [0.30, 0.20, 0.15]')
    if set(c['policy']['allowed_actions'])!=set(ACTIONS): raise ValueError('Policy action registry differs from CB18 action set')
    if not 0<=float(c['policy']['discount'])<=1: raise ValueError('policy.discount must be in [0,1]')
    for key in ['global_learning_rate','context_learning_rate','task_learning_rate']:
        x=float(c['policy'][key])
        if x<=0 or x>1: raise ValueError(f'policy.{key} must be in (0,1]')
    weight_keys=['global_weight','hc_weight','hd_weight','ot_weight','task_weight']
    if any(float(c['policy'][k])<0 for k in weight_keys) or sum(float(c['policy'][k]) for k in weight_keys)<=0:
        raise ValueError('Policy component weights must be nonnegative with a positive total')
    if float(c['policy']['max_effective_epsilon'])>1: raise ValueError('policy.max_effective_epsilon must be <= 1')
    if int(c['policy']['hard_block_after_nonpositive_repeats'])<1: raise ValueError('policy.hard_block_after_nonpositive_repeats must be >= 1')
    if not c['targets'] or c['targets'][0]['id']!='ChemDFM': raise ValueError('CB18 initial experiment expects ChemDFM as target')
    for key in ['model','backend','cache_dir','offload_folder']:
        if not str(c['targets'][0].get(key,'')).strip(): raise ValueError(f'Target missing {key}')

    attack=c['roles'].get('attack_llm')
    if not attack: raise ValueError('roles.attack_llm is missing')
    if attack.get('provider')!='openai_responses': raise ValueError('roles.attack_llm.provider must be openai_responses')
    if str(attack.get('model'))!='gpt-5.6-sol': raise ValueError('CB18 attack LLM is locked to gpt-5.6-sol')
    if str(attack.get('reasoning_effort','low')) not in {'none','low','medium','high','xhigh','max'}: raise ValueError('Invalid attack_llm reasoning_effort')
    for key in ['attempts','max_output_tokens','retry_max_output_tokens']:
        if int(attack.get(key,0))<1: raise ValueError(f'roles.attack_llm.{key} must be positive')
    if int(attack['retry_max_output_tokens'])<int(attack['max_output_tokens']): raise ValueError('attack_llm retry_max_output_tokens must be >= max_output_tokens')

    judge=c['roles'].get('judge_llm')
    if not judge: raise ValueError('roles.judge_llm is missing')
    if judge.get('provider')!='vertex_genai': raise ValueError('roles.judge_llm.provider must be vertex_genai')
    if str(judge.get('model'))!='gemini-2.5-flash': raise ValueError('CB18 judge LLM is locked to gemini-2.5-flash')
    for key in ['location','model']:
        if not str(judge.get(key,'')).strip(): raise ValueError(f'roles.judge_llm.{key} is required')
    if int(judge.get('attempts',0))<1 or int(judge.get('max_output_tokens',0))<128: raise ValueError('Invalid judge LLM retry/output settings')
    if int(judge.get('retry_max_output_tokens',0))<int(judge.get('max_output_tokens',0)): raise ValueError('judge_llm retry_max_output_tokens must be >= max_output_tokens')

    ev=c['evidence']
    ranking_weights=['reliability_weight','support_weight','reward_weight','q_weight']
    if any(float(ev.get(k,0))<0 for k in ranking_weights) or sum(float(ev.get(k,0)) for k in ranking_weights)<=0:
        raise ValueError('Evidence ranking weights must be nonnegative with a positive total')
    if int(ev.get('target_support_attempts',0))<1: raise ValueError('evidence.target_support_attempts must be >= 1')
    if float(ev.get('wilson_z',0))<=0: raise ValueError('evidence.wilson_z must be > 0')
    if not bool(ev.get('require_task_state_match_for_replay',False)): raise ValueError('CB18 requires task-state-compatible evidence replay')
    if ev.get('optimized_primary')!='exact_replay': raise ValueError('CB18 optimized_primary is locked to exact_replay')
    if ev.get('optimized_fallback')!='frozen_q_fresh_realization': raise ValueError('CB18 optimized_fallback is locked to frozen_q_fresh_realization')
    for key in ['training_artifact_path','frozen_artifact_path','rankings_artifact_path','freeze_snapshot_path']:
        if not str(ev.get(key,'')).strip(): raise ValueError(f'evidence.{key} is required')

    th=c['thresholds']
    for key in ['goal_progress_success','task_fidelity_min','chemistry_relevance_min']:
        if not 0<=float(th[key])<=1: raise ValueError(f'thresholds.{key} must be in [0,1]')
    for key in ['policy_violation_min','actionable_harm_min']:
        if not 0<=int(th[key])<=4: raise ValueError(f'thresholds.{key} must be in [0,4]')
