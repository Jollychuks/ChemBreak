#!/usr/bin/env python
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'src'
if str(SRC) not in sys.path:sys.path.insert(0,str(SRC))
from chembreak22.preflight import run_preflight

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default=str(ROOT/'configs/config.cb22.yaml')); ap.add_argument('--probe-tokenizers',action='store_true'); ap.add_argument('--probe-roles',action='store_true'); a=ap.parse_args(); print(json.dumps(run_preflight(a.config,a.probe_tokenizers,a.probe_roles),indent=2,sort_keys=True))
if __name__=='__main__':main()
