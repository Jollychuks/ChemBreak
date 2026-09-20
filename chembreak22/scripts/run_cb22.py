#!/usr/bin/env python
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'src'
if str(SRC) not in sys.path:sys.path.insert(0,str(SRC))
from chembreak22.runner import ChemBreak22Runner

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default=str(ROOT/'configs/config.cb22.yaml')); ap.add_argument('--target',choices=['ChemDFM','ChemLLM'],required=True); args=ap.parse_args()
    r=ChemBreak22Runner(args.config,args.target)
    try:print(json.dumps(r.run_all(),indent=2,sort_keys=True))
    finally:r.close()
if __name__=='__main__':main()
