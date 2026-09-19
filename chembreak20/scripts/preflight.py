import argparse,json
from chembreak20.preflight import run_preflight
p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/config.cb20.yaml'); p.add_argument('--probe-tokenizers',action='store_true'); p.add_argument('--probe-roles',action='store_true'); a=p.parse_args(); print(json.dumps(run_preflight(a.config,a.probe_tokenizers,a.probe_roles),indent=2,sort_keys=True))
