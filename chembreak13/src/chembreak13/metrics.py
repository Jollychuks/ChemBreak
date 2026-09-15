from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

def export(store, out_dir):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    ep=pd.DataFrame(store.episodes()); tr=pd.DataFrame(store.turns())
    ep.to_csv(out/'episodes.csv',index=False); tr.to_csv(out/'turns.csv',index=False)
    summary={}
    if not ep.empty:
        def stats(name,mask):
            x=ep[mask & (ep.status=='complete')]; return {'episodes':int(len(x)),'successes':int(x.success.sum()) if len(x) else 0,'asr':float(x.success.mean()) if len(x) else 0.0,'mean_turns':float(x.turns.mean()) if len(x) else 0.0}
        summary['baseline']=stats('baseline',ep.phase=='baseline')
        for e in (1,2,3): summary[f'learning_epoch_{e}']=stats(f'e{e}',(ep.phase=='learning')&(ep.epoch==e))
        summary['optimized']=stats('optimized',ep.phase=='optimized')
        summary['delta_asr_percentage_points']=100*(summary['optimized']['asr']-summary['baseline']['asr'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True))
    if not tr.empty:
        pol=tr[tr.action_id!='BASELINE_ORIGINAL'].groupby(['phase','epoch','action_id'],dropna=False).agg(uses=('action_id','size'),mean_reward=('reward','mean')).reset_index()
        pol.to_csv(out/'strategy_summary.csv',index=False)
    return summary
