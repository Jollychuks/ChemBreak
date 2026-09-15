from pathlib import Path
import yaml
from chembreak16.runner import ChemBreak16Runner

def config(tmp_path,train=2,hold=1):
    root=Path(__file__).resolve().parents[1]; c=yaml.safe_load((root/'configs/config.cb16.yaml').read_text()); r=c['run']; r.update({'project_root':str(root),'task_bank_path':str(root/'data/final_task_bank.csv'),'partition_manifest_path':str(root/'data/CB12_partition_manifest_v1.csv'),'partition_lock_path':str(root/'data/CB12_partition_lock_v1.json'),'train_manifest_path':str(root/'data/CB16_train24_manifest_v1.csv'),'holdout_manifest_path':str(root/'data/CB16_holdout12_manifest_v1.csv'),'selection_lock_path':str(root/'data/CB16_selection_lock_v1.json'),'output_root':str(tmp_path/'runs'),'dry_run':True,'train_task_limit':train,'holdout_task_limit':hold,'live_progress':False}); c['policy']['training_artifact_path']=str(tmp_path/'policy/train.json'); c['policy']['frozen_artifact_path']=str(tmp_path/'policy/frozen.json'); p=tmp_path/'c.yaml'; p.write_text(yaml.safe_dump(c,sort_keys=False)); return p

def test_end_to_end_mock_includes_unseen_holdout(tmp_path):
    r=ChemBreak16Runner(config(tmp_path))
    try:
        initial=r.run_train_baseline(); assert initial['train_optimized']['asr'] is None and initial['holdout_optimized']['asr'] is None; r.run_learning(); r.freeze_policy(); r.run_train_optimized(); r.run_holdout_baseline(); s=r.run_holdout_optimized()
        assert s['train_baseline']['episodes']==2 and s['train_optimized']['episodes']==2 and s['holdout_baseline']['episodes']==1 and s['holdout_optimized']['episodes']==1
        assert s['holdout_delta_asr_percentage_points'] is not None
    finally:r.close()

def test_holdout_firewall_refuses_access_before_freeze(tmp_path):
    r=ChemBreak16Runner(config(tmp_path,1,1))
    try:
        try:r.run_holdout_baseline()
        except FileNotFoundError: pass
        else: raise AssertionError('Holdout was accessible before policy freeze')
    finally:r.close()
