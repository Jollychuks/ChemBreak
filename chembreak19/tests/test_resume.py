from .test_dry_run import config
from chembreak19.runner import ChemBreak19Runner

def test_resume_does_not_duplicate(tmp_path):
    p=config(tmp_path,2); r=ChemBreak19Runner(p)
    try:r.run_baseline(); r.run_learning(); r.freeze_policy(); r.run_optimized(); e1=len(r.store.episodes()); t1=len(r.store.turns()); u1=r.policy.updates; records=r.evidence.records
    finally:r.close()
    r=ChemBreak19Runner(p)
    try:r.run_baseline(); r.run_learning(); r.freeze_policy(); r.run_optimized(); assert len(r.store.episodes())==e1 and len(r.store.turns())==t1 and r.policy.updates==u1 and r.evidence.records==records
    finally:r.close()
