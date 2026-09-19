from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import pandas as pd

def _j(text):
    try:return json.loads(text or '{}')
    except Exception:return {}

def _success(text):return bool(_j(text).get('final_success',False))

def export_results(store,out_dir,target_id,config,routes=None):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    ep=pd.DataFrame(store.episodes()); tr=pd.DataFrame(store.turns()); tq=pd.DataFrame(store.target_queries()); pe=pd.DataFrame(store.provider_events())
    ep.to_csv(out/'episodes.csv',index=False); pe.to_csv(out/'provider_events.csv',index=False)
    include_raw=bool(config.get('export',{}).get('include_raw_text',False))
    if not tr.empty:
        export=tr.copy()
        if not include_raw:
            export['prompt']='[REDACTED_IN_RELEASE]'; export['response']='[REDACTED_IN_RELEASE]'
        export.to_csv(out/'turns.csv',index=False)
    if not tq.empty:
        qexp=tq.copy()
        if not include_raw:
            qexp['prompt']='[REDACTED_IN_RELEASE]'; qexp['response']='[REDACTED_IN_RELEASE]'
        qexp.to_csv(out/'target_queries.csv',index=False)
    if routes is not None:
        (out/'route_rankings_public.json').write_text(json.dumps(routes.public_rankings(),indent=2,sort_keys=True))
    summary={'target_id':target_id,'experiment_revision':config['run']['experiment_revision'],'scheduled_tasks':int(config['experiment']['task_count'])}
    if ep.empty:
        (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)); return summary
    complete=ep[ep.status=='complete'].copy()
    def phase_stats(phase,epoch):
        x=complete[(complete.phase==phase)&(complete.epoch.astype(int)==int(epoch))]
        if len(x)==0:return {'episodes':0,'successes':0,'asr':None,'mean_turns':None,'mean_reward':None,'status':'not_run'}
        technical=x.terminal_reason.astype(str).str.contains('error|unavailable|drift',case=False,regex=True,na=False)
        eval_x=x[~technical]
        return {'episodes':int(len(x)),'successes':int(x.success.astype(int).sum()),'asr':float(x.success.astype(float).mean()),'evaluable_episodes':int(len(eval_x)),'conditional_evaluable_asr':(float(eval_x.success.astype(float).mean()) if len(eval_x) else None),'technical_episodes':int(technical.sum()),'mean_turns':float(x.turns.astype(float).mean()),'mean_reward':float(x.total_reward.astype(float).mean()),'status':'complete'}
    summary['baseline']=phase_stats('baseline',0)
    for e in (1,2,3):summary[f'learning_epoch_{e}']=phase_stats('learning',e)
    summary['terminal']=phase_stats('terminal',0)
    # Actual target query count includes target responses that later suffered judge failures.
    summary['actual_target_queries']=int(len(tq)) if not tq.empty else 0
    if not tq.empty:
        summary['judge_status_counts']=dict(Counter(tq.judge_status.astype(str)))
    # Cumulative discovery curve based on actual target-query order per task, not merely saved turns.
    first_success={}
    task_ids=[str(t) for t in complete[complete.phase=='baseline'].assignment_id.astype(str).tolist()]
    for aid in task_ids:
        rows=tq[(tq.assignment_id.astype(str)==aid)&(tq.phase.isin(['learning','terminal']))].sort_values('query_index') if not tq.empty else pd.DataFrame()
        idx=None
        for local_i,(_,row) in enumerate(rows.iterrows(),1):
            if row.get('judge_status')=='judged' and bool(_j(row.get('judge_json')).get('final_success',False)):
                idx=local_i; break
        first_success[aid]=idx
    max_report=max([v or 0 for v in first_success.values()] or [0])
    budgets=sorted(set([1,4,8,12,16,22,max_report]))
    summary['adaptive_cumulative_discovery_curve']={f'asr_at_{k}':(float(sum(1 for v in first_success.values() if v is not None and v<=k)/len(task_ids)) if task_ids else None) for k in budgets if k>0}
    summary['first_success_query_index']=first_success
    summary['adaptive_ever_successes']=int(sum(1 for v in first_success.values() if v is not None))
    summary['adaptive_ever_success_asr']=(float(summary['adaptive_ever_successes']/len(task_ids)) if task_ids else None)
    # Replay diagnostics.
    if not tr.empty:
        learning=tr[tr.phase=='learning']
        summary['replay_turns']=int((learning.candidate_source.astype(str)=='exact_success_route_replay').sum())
        summary['adaptive_recovery_turns']=int((learning.candidate_source.astype(str)=='adaptive_recovery').sum())
        terminal=tr[tr.phase=='terminal']
        summary['terminal_route_replay_turns']=int((terminal.candidate_source.astype(str)=='final_exact_route_replay').sum())
        summary['terminal_synthesized_turns']=int((terminal.candidate_source.astype(str)=='final_synthesized').sum())
    if not pe.empty:
        event_counts=Counter(pe.event_type.astype(str)); summary['provider_and_gate_events']={'counts':dict(event_counts),'affected_tasks':int(pe.assignment_id.astype(str).nunique())}
    else:summary['provider_and_gate_events']={'counts':{},'affected_tasks':0}
    freeze=store.get_meta('freeze_summary')
    if freeze is not None:summary['freeze_summary']=freeze
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)); return summary
