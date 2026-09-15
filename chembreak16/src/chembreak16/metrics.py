from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import pandas as pd

def export(store,out_dir):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    ep=pd.DataFrame(store.episodes()); tr=pd.DataFrame(store.turns())
    ep.to_csv(out/'episodes.csv',index=False); tr.to_csv(out/'turns.csv',index=False)
    summary={}
    if not ep.empty:
        def stats(mask):
            x=ep[mask & (ep.status=='complete')]
            if len(x)==0: return {'episodes':0,'successes':0,'asr':None,'mean_turns':None,'mean_reward':None,'status':'not_run'}
            return {'episodes':int(len(x)),'successes':int(x.success.sum()),'asr':float(x.success.mean()),'mean_turns':float(x.turns.mean()),'mean_reward':float(x.total_reward.mean()),'status':'complete'}
        summary['baseline']=stats(ep.phase=='baseline')
        for e in (1,2,3): summary[f'learning_epoch_{e}']=stats((ep.phase=='learning')&(ep.epoch==e))
        summary['optimized']=stats(ep.phase=='optimized')
        b=summary['baseline']['asr']; o=summary['optimized']['asr']
        summary['delta_asr_percentage_points']=None if b is None or o is None else 100*(o-b)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True))
    if not tr.empty:
        adaptive=tr[tr.action_id!='BASELINE_ORIGINAL'].copy()
        if not adaptive.empty:
            adaptive.groupby(['phase','epoch','action_id'],dropna=False).agg(uses=('action_id','size'),mean_reward=('reward','mean')).reset_index().to_csv(out/'strategy_summary.csv',index=False)
            rows=[]; support={}
            for _,r in adaptive.iterrows():
                try:d=json.loads(r.get('decision_json') or '{}')
                except Exception:d={}
                active=list(d.get('active_components',[])); blocked=list(d.get('blocked_actions',[]))
                rows.append({'phase':r['phase'],'epoch':r['epoch'],'assignment_id':r['assignment_id'],'turn_index':r['turn_index'],'action_id':r['action_id'],'selection_mode':r['selection_mode'],'base_epsilon':d.get('base_epsilon',0.0),'effective_epsilon':d.get('effective_epsilon',0.0),'q_global':d.get('q_global',r.get('q_global',0.0)),'q_hc':d.get('q_hc',0.0),'q_hd':d.get('q_hd',0.0),'q_ot':d.get('q_ot',0.0),'q_task':d.get('q_task',r.get('q_task',0.0)),'combined_q':d.get('combined_q',r.get('combined_q',0.0)),'active_components':','.join(active),'state_support_visits':d.get('state_support_visits',0),'visits_global':d.get('visits_global',0),'visits_hc':d.get('visits_hc',0),'visits_hd':d.get('visits_hd',0),'visits_ot':d.get('visits_ot',0),'visits_task':d.get('visits_task',0),'repeat_penalty':d.get('repeat_penalty',0.0),'adjusted_score':d.get('adjusted_score',0.0),'blocked_actions':','.join(blocked),'reward':r['reward']})
            diag=pd.DataFrame(rows); diag.to_csv(out/'policy_diagnostics.csv',index=False)
            for (phase,epoch),g in diag.groupby(['phase','epoch'],dropna=False):
                key=f'{phase}:E{int(epoch)}'; modes=Counter(g.selection_mode.astype(str)); active_counts=Counter()
                for s in g.active_components.astype(str):
                    for c in [x for x in s.split(',') if x]: active_counts[c]+=1
                support[key]={'decisions':int(len(g)),'mode_counts':dict(modes),'learned_support_decisions':int((g.active_components.astype(str)!='').sum()),'learned_support_rate':float((g.active_components.astype(str)!='').mean()),'nonzero_combined_q_decisions':int((g.combined_q.astype(float).abs()>1e-12).sum()),'nonzero_combined_q_rate':float((g.combined_q.astype(float).abs()>1e-12).mean()),'mean_state_support_visits':float(g.state_support_visits.astype(float).mean()),'active_component_counts':dict(active_counts)}
            (out/'policy_support_summary.json').write_text(json.dumps(support,indent=2,sort_keys=True))
    return summary
