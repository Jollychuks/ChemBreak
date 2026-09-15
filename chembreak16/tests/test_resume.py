from pathlib import Path
from .test_dry_run import config
from chembreak16.runner import ChemBreak16Runner

def test_resume_does_not_duplicate(tmp_path):
    p=config(tmp_path,2,1); r=ChemBreak16Runner(p)
    try:r.run_train_baseline(); r.run_learning(); r.freeze_policy(); r.run_train_optimized(); r.run_holdout_baseline(); r.run_holdout_optimized(); e1=len(r.store.episodes()); t1=len(r.store.turns()); u1=r.policy.updates
    finally:r.close()
    r=ChemBreak16Runner(p)
    try:r.run_train_baseline(); r.run_learning(); r.freeze_policy(); r.run_train_optimized(); r.run_holdout_baseline(); r.run_holdout_optimized(); assert len(r.store.episodes())==e1 and len(r.store.turns())==t1 and r.policy.updates==u1
    finally:r.close()
