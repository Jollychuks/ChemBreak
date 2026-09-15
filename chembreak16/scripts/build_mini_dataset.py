from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from chembreak16.dataset import load_task_bank, load_partition_manifest
from chembreak16.selection import build_train24, build_holdout12
bank=load_task_bank(ROOT/'data/final_task_bank.csv'); part=load_partition_manifest(ROOT/'data/CB12_partition_manifest_v1.csv')
print(build_train24(bank,part).to_string(index=False)); print('\n--- HOLDOUT12 ---'); print(build_holdout12(bank,part).to_string(index=False))
