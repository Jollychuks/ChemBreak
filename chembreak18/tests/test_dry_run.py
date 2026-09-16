from pathlib import Path
import yaml
from chembreak18.runner import ChemBreak18Runner

def config(tmp_path,limit=2):
    root=Path(__file__).resolve().parents[1]; c=yaml.safe_load((root/'configs/config.cb18.yaml').read_text()); r=c['run']
    r.update({'project_root':str(root),'task_bank_path':str(root/'data/final_task_bank.csv'),'mini_manifest_path':str(root/'data/CB18_mini24_manifest_v1.csv'),'mini_lock_path':str(root/'data/CB18_mini24_lock_v1.json'),'output_root':str(tmp_path/'runs'),'dry_run':True,'task_limit':limit,'live_progress':False})
    pd=tmp_path/'policy'; c['policy']['training_artifact_path']=str(pd/'train_policy.json'); c['policy']['frozen_artifact_path']=str(pd/'frozen_policy.json')
    c['evidence']['training_artifact_path']=str(pd/'train_evidence.json'); c['evidence']['frozen_artifact_path']=str(pd/'frozen_evidence.json'); c['evidence']['rankings_artifact_path']=str(pd/'rankings.json'); c['evidence']['freeze_snapshot_path']=str(pd/'freeze_snapshot.json')
    p=tmp_path/'c.yaml'; p.write_text(yaml.safe_dump(c,sort_keys=False)); return p

def test_end_to_end_mock_has_five_phase_slots_and_frozen_artifacts(tmp_path):
    r=ChemBreak18Runner(config(tmp_path))
    try:
        initial=r.run_baseline(); assert initial['optimized']['asr'] is None
        r.run_learning(); frozen=r.freeze_policy(); s=r.run_optimized()
        assert s['baseline']['episodes']==2 and s['learning_epoch_1']['episodes']==2 and s['learning_epoch_2']['episodes']==2 and s['learning_epoch_3']['episodes']==2 and s['optimized']['episodes']==2
        assert s['delta_asr_percentage_points'] is not None and r.policy.updates>0
        assert 'policy' in frozen and 'evidence' in frozen and r.freeze_snapshot_path.exists()
        assert 'asr_at_1' in s['optimized'] and 'asr_at_4' in s['optimized']
    finally:r.close()
