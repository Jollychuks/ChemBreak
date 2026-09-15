import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from chembreak14.dataset import load_task_bank
from chembreak14.selection import build_mini24
root=Path(__file__).resolve().parents[1]
bank=load_task_bank(root/'data/final_task_bank.csv')
mini=build_mini24(bank)
print(mini[['selection_order','assignment_id','hc_id','hd_id','ot_id']].to_string(index=False))
print('HC:',mini.hc_id.value_counts().sort_index().to_dict())
print('HD:',mini.hd_id.value_counts().sort_index().to_dict())
print('OT coverage:',mini.ot_id.nunique(),'/ 15')
