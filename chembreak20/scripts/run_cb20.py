import argparse,json
from chembreak20.runner import ChemBreak20Runner
p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/config.cb20.yaml'); p.add_argument('--target',choices=['ChemDFM','ChemLLM'],required=True); p.add_argument('--phase',choices=['baseline','learning','freeze','terminal','control','all'],default='all'); a=p.parse_args(); r=ChemBreak20Runner(a.config,a.target)
try:
    if a.phase=='baseline':out=r.run_baseline()
    elif a.phase=='learning':out=r.run_learning()
    elif a.phase=='freeze':out=r.freeze()
    elif a.phase=='terminal':out=r.run_terminal()
    elif a.phase=='control':out=r.run_budget_control()
    else:out=r.run_all()
    print(json.dumps(out,indent=2,sort_keys=True))
finally:r.close()
