from pathlib import Path
import yaml
from chembreak13.runner import ChemBreak13Runner
def test_end_to_end_mock(tmp_path):
    root=Path(__file__).resolve().parents[1]; c=yaml.safe_load((root/'configs/config.cb13.yaml').read_text()); c['run']['project_root']=str(root); c['run']['task_bank_path']=str(root/'data/final_task_bank.csv'); c['run']['mini_manifest_path']=str(root/'data/CB13_mini24_manifest_v1.csv'); c['run']['mini_lock_path']=str(root/'data/CB13_mini24_lock_v1.json'); c['run']['output_root']=str(tmp_path/'runs'); c['run']['dry_run']=True; c['run']['task_limit']=2; c['policy']['training_artifact_path']=str(tmp_path/'policy/train.json'); c['policy']['frozen_artifact_path']=str(tmp_path/'policy/frozen.json'); p=tmp_path/'c.yaml'; p.write_text(yaml.safe_dump(c,sort_keys=False)); r=ChemBreak13Runner(p)
    try:
        r.run_baseline(); r.run_learning(); r.freeze_policy(); r.run_optimized(); s=r.export_results(); assert s['baseline']['episodes']==2 and s['optimized']['episodes']==2
    finally: r.close()
