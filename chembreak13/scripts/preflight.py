from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import argparse, json
from chembreak13.preflight import run_preflight
p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--probe-tokenizer',action='store_true'); p.add_argument('--probe-roles',action='store_true'); a=p.parse_args()
print(json.dumps(run_preflight(a.config,a.probe_tokenizer,a.probe_roles),indent=2,sort_keys=True))
