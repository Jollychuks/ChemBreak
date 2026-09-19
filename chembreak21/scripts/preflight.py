#!/usr/bin/env python
from __future__ import annotations
import argparse,json
from chembreak21.preflight import run_preflight

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/config.cb21.yaml'); ap.add_argument('--probe-tokenizers',action='store_true'); ap.add_argument('--probe-roles',action='store_true'); a=ap.parse_args(); print(json.dumps(run_preflight(a.config,a.probe_tokenizers,a.probe_roles),indent=2,sort_keys=True))
if __name__=='__main__':main()
