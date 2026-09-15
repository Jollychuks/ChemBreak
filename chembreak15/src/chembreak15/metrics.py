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
        def stats(mask):
            x=ep[mask & (ep.status=='complete')]
            return {'episodes':int(len(x)),'successes':int(x.success.sum()) if len(x) else 0,'asr':float(x.success.mean()) if len(x) else 0.0,'mean_turns':float(x.turns.mean()) if len(x) else 0.0,'mean_reward':float(x.total_reward.mean()) if len(x) else 0.0}
        summary['baseline']=stats(ep.phase=='baseline')
        for e in (1,2,3): summary[f'learning_epoch_{e}']=stats((ep.phase=='learning')&(ep.epoch==e))
        summary['optimized']=stats(ep.phase=='optimized')
        summary['delta_asr_percentage_points']=100*(summary['optimized']['asr']-summary['baseline']['asr'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True))
    if not tr.empty:
        adaptive=tr[tr.action_id!='BASELINE_ORIGINAL'].copy()
        if not adaptive.empty:
            pol=adaptive.groupby(['phase','epoch','action_id'],dropna=False).agg(uses=('action_id','size'),mean_reward=('reward','mean')).reset_index()
            pol.to_csv(out/'strategy_summary.csv',index=False)
            rows=[]
            for _,r in adaptive.iterrows():
                try: d=json.loads(r.get('decision_json') or '{}')
                except Exception: d={}
                rows.append({
                    'phase':r['phase'],'epoch':r['epoch'],'assignment_id':r['assignment_id'],'turn_index':r['turn_index'],'action_id':r['action_id'],
                    'selection_mode':r['selection_mode'],'base_epsilon':d.get('base_epsilon',0.0),'effective_epsilon':d.get('effective_epsilon',0.0),
                    'q_general':d.get('q_general',r.get('q_general',0.0)),'q_task_state':d.get('q_task_state',r.get('q_task',0.0)),
                    'combined_q':d.get('combined_q',r.get('combined_q',0.0)),'repeat_penalty':d.get('repeat_penalty',0.0),
                    'adjusted_score':d.get('adjusted_score',0.0),'blocked_actions':','.join(d.get('blocked_actions',[])),
                    'state_visits':d.get('state_visits',0),'reward':r['reward'],
                })
            pd.DataFrame(rows).to_csv(out/'policy_diagnostics.csv',index=False)
    return summary
