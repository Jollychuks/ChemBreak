from pathlib import Path
import yaml
from chembreak14.runner import ChemBreak14Runner

def _config(tmp_path, task_limit=2):
    root=Path(__file__).resolve().parents[1]; c=yaml.safe_load((root/'configs/config.cb14.yaml').read_text())
    c['run'].update({'project_root':str(root),'task_bank_path':str(root/'data/final_task_bank.csv'),'mini_manifest_path':str(root/'data/CB14_mini24_manifest_v1.csv'),'mini_lock_path':str(root/'data/CB14_mini24_lock_v1.json'),'output_root':str(tmp_path/'runs'),'dry_run':True,'task_limit':task_limit,'live_progress':False})
    c['policy']['training_artifact_path']=str(tmp_path/'policy/train.json'); c['policy']['frozen_artifact_path']=str(tmp_path/'policy/frozen.json')
    p=tmp_path/'c.yaml'; p.write_text(yaml.safe_dump(c,sort_keys=False)); return p

def test_resume_does_not_duplicate_episodes_or_policy_updates(tmp_path):
    p=_config(tmp_path)
    r=ChemBreak14Runner(p)
    try: r.run_baseline(); r.run_learning(); r.freeze_policy(); r.run_optimized(); first_updates=r.policy.updates
    finally: r.close()
    r=ChemBreak14Runner(p)
    try:
        r.run_baseline(); r.run_learning(); r.freeze_policy(); r.run_optimized()
        assert r.policy.updates==first_updates==24
        assert len(r.store.episodes())==10
        assert len(r.store.turns())==34
    finally: r.close()

def test_checkpoint_identity_guard(tmp_path):
    p=_config(tmp_path,1)
    r=ChemBreak14Runner(p); r.close()
    c=yaml.safe_load(p.read_text()); c['targets'][0]['model']='Different/Model'; q=tmp_path/'changed.yaml'; q.write_text(yaml.safe_dump(c,sort_keys=False))
    try:
        ChemBreak14Runner(q)
    except RuntimeError as exc:
        assert 'different experiment identity' in str(exc)
    else:
        raise AssertionError('Identity mismatch was not rejected')
