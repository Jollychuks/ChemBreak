from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import argparse
from chembreak19.runner import ChemBreak19Runner
p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--phase',choices=['baseline','learning','freeze','optimized','all'],default='all'); a=p.parse_args()
r=ChemBreak19Runner(a.config)
try:
    if a.phase in {'baseline','all'}: r.run_baseline()
    if a.phase in {'learning','all'}: r.run_learning()
    if a.phase in {'freeze','all'}: r.freeze_policy()
    if a.phase in {'optimized','all'}: r.run_optimized()
    print(r.export_results())
finally: r.close()
