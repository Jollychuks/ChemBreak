from pathlib import Path
import yaml
from chembreak16.runner import ChemBreak16Runner

def config(tmp_path,limit=2):
    root=Path(__file__).resolve().parents[1]; c=yaml.safe_load((root/'configs/config.cb16.yaml').read_text()); r=c['run']
    r.update({'project_root':str(root),'task_bank_path':str(root/'data/final_task_bank.csv'),'mini_manifest_path':str(root/'data/CB16_mini24_manifest_v1.csv'),'mini_lock_path':str(root/'data/CB16_mini24_lock_v1.json'),'output_root':str(tmp_path/'runs'),'dry_run':True,'task_limit':limit,'live_progress':False})
    c['policy']['training_artifact_path']=str(tmp_path/'policy/train.json'); c['policy']['frozen_artifact_path']=str(tmp_path/'policy/frozen.json')
    p=tmp_path/'c.yaml'; p.write_text(yaml.safe_dump(c,sort_keys=False)); return p

def test_end_to_end_mock_has_five_phase_slots_and_null_optimized_before_run(tmp_path):
    r=ChemBreak16Runner(config(tmp_path))
    try:
        initial=r.run_baseline(); assert initial['optimized']['asr'] is None and initial['delta_asr_percentage_points'] is None
        r.run_learning(); r.freeze_policy(); s=r.run_optimized()
        assert s['baseline']['episodes']==2 and s['learning_epoch_1']['episodes']==2 and s['learning_epoch_2']['episodes']==2 and s['learning_epoch_3']['episodes']==2 and s['optimized']['episodes']==2
        assert s['delta_asr_percentage_points'] is not None
        assert r.policy.updates>0
    finally:r.close()
