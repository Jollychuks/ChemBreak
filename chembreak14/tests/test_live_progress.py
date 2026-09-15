from pathlib import Path
import yaml
from chembreak14.runner import ChemBreak14Runner

def test_live_progress_prints_running_results(tmp_path,capsys):
    root=Path(__file__).resolve().parents[1]
    c=yaml.safe_load((root/'configs/config.cb14.yaml').read_text())
    c['run'].update({'project_root':str(root),'task_bank_path':str(root/'data/final_task_bank.csv'),'mini_manifest_path':str(root/'data/CB14_mini24_manifest_v1.csv'),'mini_lock_path':str(root/'data/CB14_mini24_lock_v1.json'),'output_root':str(tmp_path/'runs'),'dry_run':True,'task_limit':1,'live_progress':True})
    c['policy']['training_artifact_path']=str(tmp_path/'policy/train.json'); c['policy']['frozen_artifact_path']=str(tmp_path/'policy/frozen.json')
    p=tmp_path/'c.yaml'; p.write_text(yaml.safe_dump(c,sort_keys=False))
    r=ChemBreak14Runner(p)
    try:
        r.run_baseline()
    finally:
        r.close()
    out=capsys.readouterr().out
    assert 'starting 1/1' in out
    assert 'turn=1' in out
    assert 'running_ASR=' in out
    assert 'target_queries=' in out
