from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from chembreak16.preflight import run_preflight
config=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'configs/config.cb16.yaml'
print(json.dumps(run_preflight(config),indent=2,sort_keys=True))
