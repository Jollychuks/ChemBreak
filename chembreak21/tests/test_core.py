from pathlib import Path
import json,yaml
from chembreak21.config import load_config,validate_config
from chembreak21.dataset import load_prompts,load_manifest,selected_tasks
from chembreak21.selection import verify_bundle
from chembreak21.constants import TASK_COUNT,SOURCE_PROMPTS_SHA256
from chembreak21.utils import sha256_file

ROOT=Path(__file__).resolve().parents[1]

def test_dataset_is_locked_28_unique():
    df=load_prompts(ROOT/'data/prompts.csv')
    assert len(df)==TASK_COUNT==28
    assert df.Behavior.str.strip().str.casefold().nunique()==28
    assert sha256_file(ROOT/'data/prompts.csv')==SOURCE_PROMPTS_SHA256

def test_manifest_and_selected_tasks():
    m=load_manifest(ROOT/'data/CB21_prompts28_manifest_v1.csv')
    t=selected_tasks(ROOT/'data/prompts.csv',ROOT/'data/CB21_prompts28_manifest_v1.csv')
    assert len(m)==len(t)==28
    assert t.assignment_id.iloc[0]=='CB21P-0001' and t.assignment_id.iloc[-1]=='CB21P-0028'
    assert t.original_prompt.equals(t.goal_intent_anchor)

def test_bundle_lock():
    out=verify_bundle(ROOT/'data/prompts.csv',ROOT/'data/CB21_prompts28_manifest_v1.csv',ROOT/'data/CB21_prompts28_lock_v1.json')
    assert out['status']=='ok' and out['tasks']==28

def test_config_locked_models_and_epochs():
    c=load_config(ROOT/'configs/config.cb21.yaml'); assert validate_config(c)
    assert c['roles']['attack_llm']['model']=='gemini-3.1-pro-preview'
    assert c['roles']['judge_llm']['model']=='gemini-3.8-flash'
    assert c['experiment']['epoch_epsilons']==[.30,.20,.15]
    assert c['experiment']['max_turns']==4

def test_new_dataset_removes_old_taxonomy_layers():
    c=load_config(ROOT/'configs/config.cb21.yaml')
    assert 'hc_weight' not in c['policy'] and 'hd_weight' not in c['policy'] and 'ot_weight' not in c['policy']
