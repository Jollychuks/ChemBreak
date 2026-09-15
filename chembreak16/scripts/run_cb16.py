from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from chembreak16.runner import ChemBreak16Runner
config=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'configs/config.cb16.yaml'
r=ChemBreak16Runner(config)
try:
    r.run_train_baseline(); r.run_learning(); r.freeze_policy(); r.run_train_optimized(); r.run_holdout_baseline(); r.run_holdout_optimized(); print(r.export_results())
finally:r.close()
