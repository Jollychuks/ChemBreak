from pathlib import Path
import json
import pytest
from chembreak20.constants import ACTIONS,FROZEN_ASSIGNMENT_IDS,NAMESPACE
from chembreak20.config import load_config,validate_config
from chembreak20.dataset import selected_tasks
from chembreak20.selection import verify_bundle
from chembreak20.state import EpisodeState
from chembreak20.prompts import task_lock,attack_prompt

ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/'configs/config.cb20.yaml'

def test_namespace():assert NAMESPACE=='CB20'
def test_actions_unique():assert len(ACTIONS)==len(set(ACTIONS))==6
def test_config_valid():assert validate_config(load_config(CFG)) is True
def test_model_roles_locked():
    c=load_config(CFG); assert c['roles']['attack_llm']['model']=='gpt-5.6-sol'; assert c['roles']['judge_llm']['model']=='gemini-3.8-flash'
def test_targets_locked():
    c=load_config(CFG); assert [x['id'] for x in c['targets']]==['ChemDFM','ChemLLM']
def test_panel_bundle():
    c=load_config(CFG); r=c['run']; x=verify_bundle(r['task_bank_path'],r['mini_manifest_path'],r['mini_lock_path']); assert x['tasks']==24 and x['ot_coverage']==15
def test_panel_exact_ids():
    c=load_config(CFG); d=selected_tasks(c['run']['task_bank_path'],c['run']['mini_manifest_path']); assert tuple(d.assignment_id.astype(str))==FROZEN_ASSIGNMENT_IDS
def test_task_lock_source_derived():
    c=load_config(CFG); row=selected_tasks(c['run']['task_bank_path'],c['run']['mini_manifest_path']).iloc[0].to_dict(); lock=task_lock(row); assert lock['original_prompt']==row['benchmark_prompt']; assert lock['original_goal']==row['main_goal']; assert lock['original_behavioral_intent']==row['main_goal']; assert lock['requested_outcome']==row['output_type']
def test_episode_initial_is_fresh():
    c=load_config(CFG); row=selected_tasks(c['run']['task_bank_path'],c['run']['mini_manifest_path']).iloc[0].to_dict(); s=EpisodeState.initial(row,4); assert s.turn_index==0 and s.progress==0 and s.history==[] and s.route_actions==[]
def test_attack_prompt_contains_immutable_lock_not_new_fields():
    c=load_config(CFG); row=selected_tasks(c['run']['task_bank_path'],c['run']['mini_manifest_path']).iloc[0].to_dict(); s=EpisodeState.initial(row,4); payload=json.loads(attack_prompt(row,'REPHRASE_GOAL',s)); assert payload['task_lock']['original_prompt']==row['benchmark_prompt']; assert payload['selected_action']=='REPHRASE_GOAL'
