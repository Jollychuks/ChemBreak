#!/usr/bin/env python
from __future__ import annotations
import argparse,json
from chembreak21.runner import ChemBreak21Runner

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/config.cb21.yaml'); ap.add_argument('--target',choices=['ChemDFM','ChemLLM'],required=True); args=ap.parse_args()
    r=ChemBreak21Runner(args.config,args.target)
    try:print(json.dumps(r.run_all(),indent=2,sort_keys=True))
    finally:r.close()
if __name__=='__main__':main()
