from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

def export(store,out_dir):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); ep=pd.DataFrame(store.episodes()); tr=pd.DataFrame(store.turns()); ep.to_csv(out/'episodes.csv',index=False); tr.to_csv(out/'turns.csv',index=False)
    def stats(phase,epoch=None):
        if ep.empty: return {'status':'not_run','episodes':0,'successes':0,'asr':None,'mean_turns':None,'mean_reward':None}
        m=(ep.phase==phase)&(ep.status=='complete')
        if epoch is not None: m=m&(ep.epoch==epoch)
        x=ep[m]
        if x.empty:return {'status':'not_run','episodes':0,'successes':0,'asr':None,'mean_turns':None,'mean_reward':None}
        return {'status':'complete','episodes':int(len(x)),'successes':int(x.success.sum()),'asr':float(x.success.mean()),'mean_turns':float(x.turns.mean()),'mean_reward':float(x.total_reward.mean())}
    summary={'train_baseline':stats('train_baseline')}
    for e in (1,2,3): summary[f'learning_epoch_{e}']=stats('learning',e)
    summary['train_optimized']=stats('train_optimized'); summary['holdout_baseline']=stats('holdout_baseline'); summary['holdout_optimized']=stats('holdout_optimized')
    def delta(a,b):
        return None if summary[a]['asr'] is None or summary[b]['asr'] is None else 100*(summary[b]['asr']-summary[a]['asr'])
    summary['train_delta_asr_percentage_points']=delta('train_baseline','train_optimized'); summary['holdout_delta_asr_percentage_points']=delta('holdout_baseline','holdout_optimized')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True))
    if not tr.empty:
        adaptive=tr[tr.action_id!='BASELINE_ORIGINAL'].copy()
        if not adaptive.empty:
            adaptive.groupby(['phase','epoch','action_id'],dropna=False).agg(uses=('action_id','size'),mean_reward=('reward','mean')).reset_index().to_csv(out/'strategy_summary.csv',index=False)
            rows=[]
            for _,r in adaptive.iterrows():
                try:d=json.loads(r.get('decision_json') or '{}')
                except Exception:d={}
                rows.append({'phase':r['phase'],'epoch':r['epoch'],'assignment_id':r['assignment_id'],'turn_index':r['turn_index'],'action_id':r['action_id'],'selection_mode':r['selection_mode'],'base_epsilon':d.get('base_epsilon',0.0),'effective_epsilon':d.get('effective_epsilon',0.0),'q_global':d.get('q_global',0.0),'q_hc':d.get('q_hc',0.0),'q_hd':d.get('q_hd',0.0),'q_ot':d.get('q_ot',0.0),'q_task':d.get('q_task',0.0),'combined_q':d.get('combined_q',r.get('combined_q',0.0)),'active_components':','.join(d.get('active_components',[])),'state_support_visits':d.get('state_support_visits',0),'visits_global':d.get('visits_global',0),'visits_hc':d.get('visits_hc',0),'visits_hd':d.get('visits_hd',0),'visits_ot':d.get('visits_ot',0),'visits_task':d.get('visits_task',0),'repeat_penalty':d.get('repeat_penalty',0.0),'adjusted_score':d.get('adjusted_score',0.0),'blocked_actions':','.join(d.get('blocked_actions',[])),'reward':r['reward']})
            pd.DataFrame(rows).to_csv(out/'policy_diagnostics.csv',index=False)
    return summary
