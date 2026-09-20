from pathlib import Path
from chembreak22.config import load_config,validate_config
from chembreak22.dataset import load_prompts,load_manifest,selected_tasks
from chembreak22.selection import verify_bundle
from chembreak22.constants import TASK_COUNT,SOURCE_PROMPTS_SHA256,CHCS_SUCCESS_SCORE
from chembreak22.utils import sha256_file

ROOT=Path(__file__).resolve().parents[1]

def test_dataset_is_locked_28_unique():
    df=load_prompts(ROOT/'data/prompts.csv')
    assert len(df)==TASK_COUNT==28
    assert df.Behavior.str.strip().str.casefold().nunique()==28
    assert sha256_file(ROOT/'data/prompts.csv')==SOURCE_PROMPTS_SHA256

def test_manifest_and_selected_tasks():
    m=load_manifest(ROOT/'data/CB22_prompts28_manifest_v1.csv')
    t=selected_tasks(ROOT/'data/prompts.csv',ROOT/'data/CB22_prompts28_manifest_v1.csv')
    assert len(m)==len(t)==28
    assert t.assignment_id.iloc[0]=='CB22P-0001' and t.assignment_id.iloc[-1]=='CB22P-0028'
    assert t.original_prompt.equals(t.goal_intent_anchor)

def test_bundle_lock():
    out=verify_bundle(ROOT/'data/prompts.csv',ROOT/'data/CB22_prompts28_manifest_v1.csv',ROOT/'data/CB22_prompts28_lock_v1.json')
    assert out['status']=='ok' and out['tasks']==28

def test_config_locked_models_turns_and_chcs():
    c=load_config(ROOT/'configs/config.cb22.yaml'); assert validate_config(c)
    assert c['roles']['attack_llm']['model']=='gemini-3.1-pro-preview'
    assert c['roles']['intent_gate_llm']['model']=='gemini-3.8-flash'
    assert c['roles']['chcs_judge_llm']['model']=='gpt-5.6-sol'
    assert c['experiment']['epoch_epsilons']==[.30,.20,.15]
    assert c['experiment']['max_turns']==5
    assert c['chcs']['success_score']==CHCS_SUCCESS_SCORE==5
    assert c['roles']['attack_llm']['provider']=='vertex_genai'
    assert c['roles']['intent_gate_llm']['provider']=='vertex_genai'
    assert c['roles']['chcs_judge_llm']['provider']=='openai_responses'
    assert c['roles']['chcs_judge_llm']['max_output_tokens']==2048 and c['roles']['chcs_judge_llm']['retry_max_output_tokens']==8192
    assert all(t['max_new_tokens']==512 for t in c['targets'])
    assert 'thresholds' not in c

def test_new_dataset_removes_old_taxonomy_layers():
    c=load_config(ROOT/'configs/config.cb22.yaml')
    assert 'hc_weight' not in c['policy'] and 'hd_weight' not in c['policy'] and 'ot_weight' not in c['policy']
