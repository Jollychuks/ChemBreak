import yaml
from chembreak16.runner import ChemBreak16Runner
from .test_dry_run import config

def test_live_progress_has_hierarchical_diagnostics(tmp_path,capsys):
    p=config(tmp_path,1); c=yaml.safe_load(p.read_text()); c['run']['live_progress']=True; p.write_text(yaml.safe_dump(c,sort_keys=False)); r=ChemBreak16Runner(p)
    try:r.run_baseline(); r.run_learning()
    finally:r.close()
    out=capsys.readouterr().out
    for term in ['running_ASR=','Qg=','Qhc=','Qhd=','Qot=','Qt=','active=','support=','mode=']:
        assert term in out
