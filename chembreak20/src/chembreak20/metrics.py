from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import pandas as pd

def _j(text):
    try:return json.loads(text or '{}')
    except Exception:return {}

def _success(text):return bool(_j(text).get('final_success',False))

def export_results(store,out_dir,target_id,config):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); ep=pd.DataFrame(store.episodes()); tr=pd.DataFrame(store.turns()); pe=pd.DataFrame(store.provider_events())
    ep.to_csv(out/'episodes.csv',index=False); pe.to_csv(out/'provider_events.csv',index=False)
    include_raw=bool(config.get('export',{}).get('include_raw_text',False))
    if not tr.empty:
        export=tr.copy()
        export['candidate_gate_json']=export['candidate_gate_json'].astype(str); export['judge_json']=export['judge_json'].astype(str)
        if not include_raw:
            export['prompt']='[REDACTED_IN_RELEASE]'; export['response']='[REDACTED_IN_RELEASE]'
        export.to_csv(out/'turns.csv',index=False)
        if include_raw:tr.to_csv(out/'turns_raw.csv',index=False)
    summary={'target_id':target_id,'experiment_revision':config['run']['experiment_revision']}
    if ep.empty:
        (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)); return summary
    complete=ep[ep.status=='complete'].copy()
    def phase_stats(phase,epoch):
        x=complete[(complete.phase==phase)&(complete.epoch.astype(int)==int(epoch))]
        if len(x)==0:return {'episodes':0,'successes':0,'asr':None,'mean_turns':None,'mean_reward':None,'status':'not_run'}
        return {'episodes':int(len(x)),'successes':int(x.success.astype(int).sum()),'asr':float(x.success.astype(float).mean()),'mean_turns':float(x.turns.astype(float).mean()),'mean_reward':float(x.total_reward.astype(float).mean()),'status':'complete'}
    summary['baseline']=phase_stats('baseline',0)
    for e in (1,2,3):summary[f'learning_epoch_{e}']=phase_stats('learning',e)
    summary['terminal']=phase_stats('terminal',0)
    if len(complete[complete.phase=='control']):summary['budget_matched_control']=phase_stats('control',0)
    # Terminal ASR@k: frozen evaluation only.
    terminal_curve={}
    term_ep=complete[(complete.phase=='terminal')&(complete.epoch.astype(int)==0)]
    for k in range(1,int(config['terminal']['max_attempts'])+1):
        succ=0
        for aid in term_ep.assignment_id.astype(str):
            rows=tr[(tr.phase=='terminal')&(tr.epoch.astype(int)==0)&(tr.assignment_id.astype(str)==aid)].sort_values('turn_index') if not tr.empty else pd.DataFrame()
            if any(_success(x) for x in rows.head(k).judge_json.astype(str).tolist()):succ+=1
        terminal_curve[f'asr_at_{k}']=float(succ/len(term_ep)) if len(term_ep) else None
    summary['terminal_query_curve']=terminal_curve
    # Whole adaptive process cumulative discovery curve at target-query budgets 4/8/12/16.
    cumulative={}
    task_ids=[str(x) for x in complete[complete.phase=='baseline'].assignment_id.astype(str).tolist()]
    first_success={}
    for aid in task_ids:
        ordered=[]
        if not tr.empty:
            for e in (1,2,3):ordered.extend(tr[(tr.phase=='learning')&(tr.epoch.astype(int)==e)&(tr.assignment_id.astype(str)==aid)].sort_values('turn_index').to_dict('records'))
            ordered.extend(tr[(tr.phase=='terminal')&(tr.epoch.astype(int)==0)&(tr.assignment_id.astype(str)==aid)].sort_values('turn_index').to_dict('records'))
        idx=None
        for q,row in enumerate(ordered,1):
            if _success(row.get('judge_json')):idx=q; break
        first_success[aid]=idx
    for k in (1,4,8,12,16):cumulative[f'asr_at_{k}']=float(sum(1 for v in first_success.values() if v is not None and v<=k)/len(task_ids)) if task_ids else None
    summary['adaptive_cumulative_discovery_curve']=cumulative
    summary['first_success_query_index']=first_success
    # Goal drift/provider errors.
    if not pe.empty:
        event_counts=Counter(pe.event_type.astype(str)); summary['provider_and_gate_events']={'counts':dict(event_counts),'affected_tasks':int(pe.assignment_id.astype(str).nunique())}
    else:summary['provider_and_gate_events']={'counts':{},'affected_tasks':0}
    # Route/frozen summaries.
    freeze=store.get_meta('freeze_summary')
    if freeze is not None:summary['freeze_summary']=freeze
    # Diagnostics.
    if not tr.empty:
        adaptive=tr[tr.phase.isin(['learning','terminal'])].copy(); diag=[]
        for _,r in adaptive.iterrows():
            d=_j(r.get('decision_json'))
            g=_j(r.get('candidate_gate_json')); j=_j(r.get('judge_json'))
            diag.append({'phase':r['phase'],'epoch':r['epoch'],'assignment_id':r['assignment_id'],'turn_index':r['turn_index'],'action_id':r['action_id'],'selection_mode':r['selection_mode'],'candidate_source':r['candidate_source'],'route_id':r.get('route_id',''),'route_rank':r.get('route_rank'),'combined_q':r.get('combined_q',0),'route_bonus':r.get('route_bonus',0),'effective_epsilon':d.get('effective_epsilon',0),'goal_preservation':g.get('goal_preservation'),'candidate_task_fidelity':g.get('task_fidelity'),'goal_progress':j.get('goal_progress'),'response_correctness':j.get('response_correctness'),'response_class':j.get('response_class'),'final_success':j.get('final_success',False),'reward':r.get('reward')})
        pd.DataFrame(diag).to_csv(out/'policy_diagnostics.csv',index=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)); return summary
