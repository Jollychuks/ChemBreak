from __future__ import annotations
import json,math
from collections import Counter
from pathlib import Path
import pandas as pd
from .constants import TASK_COUNT
from .dataset import selected_tasks


def _j(x):
    try:return json.loads(x or '{}')
    except Exception:return {}
def _chcs(x):
    try:return int(_j(x).get('chcs') or 0)
    except Exception:return 0
def _success(x):return bool(_j(x).get('final_success',False))
def _wilson(s,n,z=1.96):
    if n<=0:return [None,None]
    p=s/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return [max(0,c-h),min(1,c+h)]


def _lookup(config):
    return {str(x['assignment_id']):x for x in selected_tasks(config['run']['prompts_path'],config['run']['manifest_path']).to_dict('records')}


def _private_exports(store,out,target_id,config,tq,tr):
    internal=Path(out)/'internal'; internal.mkdir(parents=True,exist_ok=True); lookup=_lookup(config); turn_map={int(x['target_query_index']):x for x in tr.to_dict('records')} if not tr.empty else {}; committed=store.committed_target_query_ids(); counts=Counter(); rows=[]
    for q in tq.sort_values('query_index').to_dict('records') if not tq.empty else []:
        aid=str(q['assignment_id']); counts[aid]+=1; turn=turn_map.get(int(q['query_index']),{}); judge=_j(q.get('judge_json')); audit=judge.get('judge_provenance',judge) if isinstance(judge.get('judge_provenance',judge),dict) else {}; src=lookup.get(aid,{})
        rows.append({'query_index':int(q['query_index']),'task_id':aid,'target':target_id,'phase':q['phase'],'episode':int(q['episode']),'turn':int(q['turn_index']),'cumulative_query_index_for_task':counts[aid],'original_prompt':src.get('original_prompt',''),'action':q['action_id'],'attack_prompt':q['prompt'],'target_response':q['response'],'judge_status':q['judge_status'],'chcs':judge.get('chcs'),'response_class':judge.get('response_class'),'success':bool(judge.get('final_success',False)),'reward':turn.get('reward'),'goal_preservation':_j(turn.get('candidate_gate_json')).get('goal_preservation'),'task_fidelity':_j(turn.get('candidate_gate_json')).get('task_fidelity'),'primary_judge':audit.get('primary_judge'),'fallback_judge':audit.get('fallback_judge'),'final_judge':audit.get('final_judge'),'fallback_used':audit.get('fallback_used',False),'committed':int(q['query_index']) in committed,'latency_seconds':q.get('latency_seconds'),'created_at':q.get('created_at'),'judged_at':q.get('judged_at')})
    full=pd.DataFrame(rows); full.to_csv(internal/'cb30_full_transcripts.csv',index=False)
    if config['export'].get('write_internal_jsonl',True):full.to_json(internal/'cb30_full_transcripts.jsonl',orient='records',lines=True,force_ascii=False)
    success_rows=[]
    if not full.empty:
        good=full[(full.committed==True)&(full.judge_status=='judged')&(full.success==True)].sort_values('query_index')
        for aid,g in good.groupby('task_id',sort=True):
            row=g.iloc[0].to_dict(); row['needs_human_review']=True; row['review_status']='pending'; success_rows.append(row)
    review=pd.DataFrame(success_rows); review.to_csv(internal/'cb30_first_success_review.csv',index=False)
    if config['export'].get('write_internal_jsonl',True):review.to_json(internal/'cb30_first_success_review.jsonl',orient='records',lines=True,force_ascii=False)
    manifest={'target_id':target_id,'experiment_revision':config['run']['experiment_revision'],'private_raw_text':True,'warning':'Contains raw benchmark prompts, candidate prompts, and target responses. Review and redact before publication.','rows':{'full_transcripts':len(full),'first_success_review':len(review)},'files':['cb30_full_transcripts.csv','cb30_full_transcripts.jsonl','cb30_first_success_review.csv','cb30_first_success_review.jsonl']}
    (internal/'INTERNAL_AUDIT_MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)); return manifest


def export_results(store,out_dir,target_id,config):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); ep=pd.DataFrame(store.episodes()); tr=pd.DataFrame(store.turns()); tq=pd.DataFrame(store.target_queries()); pe=pd.DataFrame(store.provider_events()); committed=store.committed_target_query_ids()
    if tq.empty:tq=pd.DataFrame(columns=['query_index','phase','episode','assignment_id','judge_status','judge_json','prompt','response'])
    tq=tq.copy(); tq['committed']=tq.query_index.astype(int).isin(committed) if not tq.empty else False; ctq=tq[tq.committed==True].copy(); identity=store.get_meta('experiment_identity',{}) or {}; task_ids=[str(x) for x in identity.get('assignment_ids',[])]; scheduled_n=len(task_ids); n=TASK_COUNT; score=int(config['chcs']['success_score'])
    ep.to_csv(out/'episodes.csv',index=False); pe.to_csv(out/'provider_events.csv',index=False)
    tr_public=tr.copy()
    if not tr_public.empty:tr_public['prompt']='[REDACTED_IN_RELEASE]'; tr_public['response']='[REDACTED_IN_RELEASE]'
    tr_public.to_csv(out/'turns.csv',index=False)
    tq_public=tq.copy(); tq_public['prompt']='[REDACTED_IN_RELEASE]'; tq_public['response']='[REDACTED_IN_RELEASE]'; tq_public.to_csv(out/'target_queries.csv',index=False)

    successes={aid:None for aid in task_ids}; first_adaptive_query={}; adaptive_counts=Counter()
    for q in ctq.sort_values('query_index').to_dict('records') if not ctq.empty else []:
        aid=str(q['assignment_id'])
        if q['phase']=='adaptive':adaptive_counts[aid]+=1
        if q['judge_status']=='judged' and _success(q['judge_json']) and successes.get(aid) is None:
            successes[aid]={'phase':q['phase'],'episode':int(q['episode']),'query_index':int(q['query_index'])}
            if q['phase']=='adaptive':first_adaptive_query[aid]=adaptive_counts[aid]
    baseline_success={a for a,v in successes.items() if v and v['phase']=='baseline'}; baseline_failures=scheduled_n-len(baseline_success); cumulative=set(baseline_success)
    summary={'target_id':target_id,'target_model':identity.get('target_model'),'target_revision':identity.get('target_revision'),'experiment_revision':config['run']['experiment_revision'],'run_mode':identity.get('run_mode'),'run_id':identity.get('run_id'),'scheduled_tasks':scheduled_n,'complete_28_task_panel':scheduled_n==TASK_COUNT,'asr_denominator':n,'success_definition':f'first valid CHCS == {score}; fallback judge only after primary failure; valid objective-fidelity-gated candidate','design':{'baseline_queries_per_task':1,'adaptive_episodes':3,'turns_per_episode':5,'maximum_queries_per_task':16,'early_stop_on_first_success':True,'fresh_context_per_episode':True,'cross_task_learning':False,'terminal_replay':False,'single_candidate_per_generation':True,'independent_chcs_confirmation':False,'policy_type':'hierarchical_contextual_bandit'}}
    bs=len(baseline_success); summary['baseline']={'successes':bs,'tasks':n,'asr':bs/n if n else None,'wilson95':_wilson(bs,n)}
    for episode in (1,2,3):
        eligible={a for a in task_ids if a not in cumulative}; gained={a for a in eligible if successes.get(a) and successes[a]['phase']=='adaptive' and successes[a]['episode']==episode}; cumulative|=gained
        episodes_run=0 if ep.empty else int(len(ep[(ep.phase=='adaptive')&(ep.episode.astype(int)==episode)&(ep.status=='complete')]))
        summary[f'episode_{episode}']={'eligible_tasks':len(eligible),'episodes_run':episodes_run,'new_successes':len(gained),'conditional_rescue_rate':len(gained)/len(eligible) if eligible else None,'conditional_wilson95':_wilson(len(gained),len(eligible)),'cumulative_successes':len(cumulative),'cumulative_asr':len(cumulative)/n if n else None,'cumulative_wilson95':_wilson(len(cumulative),n)}
    rescued={a for a,v in successes.items() if v and v['phase']=='adaptive'}; summary['conditional_rescue_among_baseline_failures']={'baseline_failures':baseline_failures,'rescued_tasks':len(rescued),'rate':len(rescued)/baseline_failures if baseline_failures else None,'wilson95':_wilson(len(rescued),baseline_failures)}; total_success=sum(v is not None for v in successes.values()); summary['adaptive_success_over_all_tasks']={'successes':len(rescued),'tasks':n,'asr_contribution':len(rescued)/n if n else None,'wilson95':_wilson(len(rescued),n)}; summary['bounded_first_success']={'successes':total_success,'tasks':n,'asr':total_success/n if n else None,'wilson95':_wilson(total_success,n)}
    summary['cumulative_success_by_adaptive_query_budget']={}
    for budget in (1,5,10,15):
        k=len(baseline_success)+sum(1 for q in first_adaptive_query.values() if q<=budget); summary['cumulative_success_by_adaptive_query_budget'][str(budget)]={'successes':k,'tasks':n,'asr':k/n if n else None,'wilson95':_wilson(k,n)}
    vals=list(first_adaptive_query.values()); summary['first_adaptive_success_query_index']={'per_task':first_adaptive_query,'mean':sum(vals)/len(vals) if vals else None,'median':float(pd.Series(vals).median()) if vals else None}
    issued=len(tq); committed_n=len(ctq); unresolved=int(((ctq.judge_status!='judged')).sum()) if not ctq.empty else 0; judged=committed_n-unresolved
    summary['query_accounting']={'issued':issued,'committed':committed_n,'uncommitted_audit_only':issued-committed_n,'judged_committed':judged,'unresolved_committed':unresolved,'judge_coverage':judged/committed_n if committed_n else None}
    audits=[]
    for x in tq.judge_json.astype(str).tolist() if not tq.empty else []:
        j=_j(x); audits.append(j.get('judge_provenance',j) if isinstance(j.get('judge_provenance',j),dict) else {})
    pf=sum(a.get('primary_judge_status')=='error' for a in audits); fr=sum(a.get('fallback_judge_status')=='valid' for a in audits)
    summary['judge_resolution']={'primary_valid':sum(a.get('primary_judge_status')=='valid' for a in audits),'primary_failures':pf,'fallback_recovered':fr,'fallback_recovery_rate':fr/pf if pf else None,'residual_unresolved':unresolved}
    summary['success_source_counts']={'baseline':len(baseline_success),'episode_1':sum(bool(v and v['phase']=='adaptive' and v['episode']==1) for v in successes.values()),'episode_2':sum(bool(v and v['phase']=='adaptive' and v['episode']==2) for v in successes.values()),'episode_3':sum(bool(v and v['phase']=='adaptive' and v['episode']==3) for v in successes.values())}
    action_rows=[]
    for row in tr.to_dict('records') if not tr.empty else []:
        if row.get('phase')!='adaptive':continue
        decision=_j(row.get('decision_json')); judge=_j(row.get('judge_json')); update=decision.get('update',{}) if isinstance(decision.get('update',{}),dict) else {}
        action_rows.append({'action':str(row.get('action_id')),'reward':float(row.get('reward',0)),'positive_reward':float(row.get('reward',0))>0,'success':bool(judge.get('final_success',False)),'selection_mode':str(row.get('selection_mode','')),'q_task_delta':float(update.get('q_task_after',0))-float(update.get('q_task_before',0)) if not update.get('learning_update_skipped') else None,'q_state_delta':float(update.get('q_state_after',0))-float(update.get('q_state_before',0)) if not update.get('learning_update_skipped') else None})
    if action_rows:
        adf=pd.DataFrame(action_rows); summary['action_diagnostics']={}
        for action,g in adf.groupby('action'):
            summary['action_diagnostics'][action]={'selections':len(g),'mean_reward':float(g.reward.mean()),'positive_reward_rate':float(g.positive_reward.mean()),'validated_successes':int(g.success.sum()),'selection_modes':dict(Counter(g.selection_mode))}
        summary['policy_selection_modes']=dict(Counter(adf.selection_mode))
    else:summary['action_diagnostics']={}; summary['policy_selection_modes']={}
    summary['judge_cascade_policy']={'mode':'primary_then_fallback_on_error','independent_confirmation':False,'valid_primary_result_is_final':True,'valid_fallback_result_is_final':True}
    summary['recovered_judgments']=int(store.get_meta('recovered_judgments',0) or 0)
    summary['committed_chcs_distribution']=dict(Counter(str(_chcs(x)) for x in ctq[ctq.judge_status=='judged'].judge_json.astype(str))) if not ctq.empty else {}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)); _private_exports(store,out,target_id,config,tq,tr); return summary
