from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import pandas as pd

def _judge_success(v):
    try:return bool(json.loads(v).get('success',False))
    except Exception:return False

def export(store,out_dir,max_turns=4):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    ep=pd.DataFrame(store.episodes()); tr=pd.DataFrame(store.turns()); pe=pd.DataFrame(store.provider_events())
    ep.to_csv(out/'episodes.csv',index=False); tr.to_csv(out/'turns.csv',index=False); pe.to_csv(out/'provider_events.csv',index=False)
    if not tr.empty:
        b=tr[tr.action_id.astype(str)=='BASELINE_ORIGINAL'].copy()
        if not b.empty:
            audit=[]
            for _,row in b.iterrows():
                try:j=json.loads(row.get('judge_json') or '{}')
                except Exception:j={}
                audit.append({
                    'assignment_id':str(row['assignment_id']),
                    'success':bool(j.get('success',False)),
                    'judge_claimed_success':bool(j.get('judge_claimed_success',False)),
                    'response_class':j.get('response_class'),
                    'goal_progress':j.get('goal_progress'),
                    'task_fidelity':j.get('task_fidelity'),
                    'chemistry_relevance':j.get('chemistry_relevance'),
                    'policy_violation':j.get('policy_violation'),
                    'actionable_harm':j.get('actionable_harm'),
                    'reward':row.get('reward'),
                    'prompt':row.get('prompt'),
                    'response':row.get('response'),
                })
            pd.DataFrame(audit).to_csv(out/'baseline_audit.csv',index=False)
    summary={}
    if not ep.empty:
        complete=ep[ep.status=='complete'].copy()
        def stats(mask):
            x=complete[mask.loc[complete.index] if hasattr(mask,'loc') else mask]
            if len(x)==0:return {'episodes':0,'successes':0,'asr':None,'asr_at_1':None,f'asr_at_{max_turns}':None,'mean_turns':None,'mean_reward':None,'recovery_rate':None,'target_evaluable_episodes':0,'target_evaluable_successes':0,'target_evaluable_asr':None,'provider_blocked_before_target_episodes':0,'status':'not_run'}
            ids=set((str(r.phase),int(r.epoch),str(r.assignment_id)) for _,r in x.iterrows())
            first_success=0; recovery_den=0; recovery_num=0
            for phase,epoch,aid in ids:
                turns=tr[(tr.phase==phase)&(tr.epoch==epoch)&(tr.assignment_id.astype(str)==aid)].sort_values('turn_index') if not tr.empty else pd.DataFrame()
                first=bool(_judge_success(turns.iloc[0].judge_json)) if len(turns) else False
                if first:first_success+=1
                if phase!='baseline' and len(turns) and not first:
                    recovery_den+=1
                    if bool(x[(x.phase==phase)&(x.epoch==epoch)&(x.assignment_id.astype(str)==aid)].iloc[0].success):recovery_num+=1
            final=float(x.success.mean())
            evaluable=x[x.turns.astype(int)>0]
            evaluable_asr=(float(evaluable.success.mean()) if len(evaluable) else None)
            return {'episodes':int(len(x)),'successes':int(x.success.sum()),'asr':final,'asr_at_1':float(first_success/len(x)),f'asr_at_{max_turns}':final,'mean_turns':float(x.turns.mean()),'mean_reward':float(x.total_reward.mean()),'recovery_rate':(float(recovery_num/recovery_den) if recovery_den else None),'target_evaluable_episodes':int(len(evaluable)),'target_evaluable_successes':int(evaluable.success.sum()) if len(evaluable) else 0,'target_evaluable_asr':evaluable_asr,'provider_blocked_before_target_episodes':int((x.turns.astype(int)==0).sum()),'status':'complete'}
        summary['baseline']=stats(ep.phase=='baseline')
        summary['baseline']['success_ids']=sorted(complete[(complete.phase=='baseline') & (complete.success.astype(bool))].assignment_id.astype(str).tolist())
        for e in (1,2,3):summary[f'learning_epoch_{e}']=stats((ep.phase=='learning')&(ep.epoch==e))
        summary['optimized']=stats(ep.phase=='optimized')
        b=summary['baseline']['asr']; o=summary['optimized']['asr']; summary['delta_asr_percentage_points']=None if b is None or o is None else 100*(o-b)

        def success_set(phase,epoch):
            x=complete[(complete.phase==phase)&(complete.epoch==epoch)&(complete.success.astype(bool))]
            return set(x.assignment_id.astype(str))
        s1=success_set('learning',1); s2=success_set('learning',2); s3=success_set('learning',3); so=success_set('optimized',0)
        retention={}
        prior=s1; retention['learning_epoch_2']={'eligible':len(prior),'retained':len(prior&s2),'rate':(len(prior&s2)/len(prior) if prior else None)}
        prior=s1|s2; retention['learning_epoch_3']={'eligible':len(prior),'retained':len(prior&s3),'rate':(len(prior&s3)/len(prior) if prior else None)}
        prior=s1|s2|s3; retention['optimized']={'eligible':len(prior),'retained':len(prior&so),'rate':(len(prior&so)/len(prior) if prior else None)}
        summary['success_retention']=retention
        freeze=store.get_meta('freeze_summary')
        if freeze is not None:
            summary['evidence_coverage']=freeze.get('evidence')
            summary['trajectory_coverage']=freeze.get('trajectory')
        if not pe.empty:
            blocks=pe[pe.event_type.astype(str)=='policy_block'].copy()
            grouped={}
            for (phase,epoch),g in blocks.groupby(['phase','epoch'],dropna=False):grouped[f'{phase}:E{int(epoch)}']=int(len(g))
            affected=int(blocks[['phase','epoch','assignment_id']].drop_duplicates().shape[0])
            total_adaptive=max(1,int(len(complete[complete.phase!='baseline'])))
            summary['provider_policy_blocks']={'events':int(len(blocks)),'affected_tasks':int(blocks.assignment_id.astype(str).nunique()),'affected_episodes':affected,'affected_episode_rate':float(affected/total_adaptive),'by_phase_epoch':grouped}
        else:summary['provider_policy_blocks']={'events':0,'affected_tasks':0,'affected_episodes':0,'affected_episode_rate':0.0,'by_phase_epoch':{}}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True))

    if not tr.empty:
        adaptive=tr[tr.action_id!='BASELINE_ORIGINAL'].copy()
        if not adaptive.empty:
            adaptive.groupby(['phase','epoch','action_id','candidate_source'],dropna=False).agg(uses=('action_id','size'),mean_reward=('reward','mean')).reset_index().to_csv(out/'strategy_summary.csv',index=False)
            rows=[]; support={}
            for _,r in adaptive.iterrows():
                try:d=json.loads(r.get('decision_json') or '{}')
                except Exception:d={}
                active=list(d.get('active_components',[])); blocked=list(d.get('blocked_actions',[]))
                rows.append({'phase':r['phase'],'epoch':r['epoch'],'assignment_id':r['assignment_id'],'turn_index':r['turn_index'],'action_id':r['action_id'],'candidate_id':r.get('candidate_id',''),'realization_id':r.get('realization_id',''),'candidate_source':r.get('candidate_source',''),'candidate_rank_score':r.get('candidate_rank_score'),'selection_mode':r['selection_mode'],'base_epsilon':d.get('base_epsilon',0.0),'effective_epsilon':d.get('effective_epsilon',0.0),'q_global':d.get('q_global',r.get('q_global',0.0)),'q_hc':d.get('q_hc',0.0),'q_hd':d.get('q_hd',0.0),'q_ot':d.get('q_ot',0.0),'q_task':d.get('q_task',r.get('q_task',0.0)),'combined_q':d.get('combined_q',r.get('combined_q',0.0)),'active_components':','.join(active),'state_support_visits':d.get('state_support_visits',0),'visits_global':d.get('visits_global',0),'visits_hc':d.get('visits_hc',0),'visits_hd':d.get('visits_hd',0),'visits_ot':d.get('visits_ot',0),'visits_task':d.get('visits_task',0),'confidence_global':d.get('confidence_global',0.0),'confidence_hc':d.get('confidence_hc',0.0),'confidence_hd':d.get('confidence_hd',0.0),'confidence_ot':d.get('confidence_ot',0.0),'confidence_task':d.get('confidence_task',0.0),'trajectory_id':d.get('trajectory_id',''),'trajectory_step_index':d.get('trajectory_step_index'),'trajectory_length':d.get('trajectory_length'),'trajectory_rank_score':d.get('trajectory_rank_score'),'repeat_penalty':d.get('repeat_penalty',0.0),'adjusted_score':d.get('adjusted_score',0.0),'blocked_actions':','.join(blocked),'reward':r['reward']})
            diag=pd.DataFrame(rows); diag.to_csv(out/'policy_diagnostics.csv',index=False)
            for (phase,epoch),g in diag.groupby(['phase','epoch'],dropna=False):
                key=f'{phase}:E{int(epoch)}'; modes=Counter(g.selection_mode.astype(str)); active_counts=Counter(); sources=Counter(g.candidate_source.astype(str))
                for s in g.active_components.astype(str):
                    for c in [x for x in s.split(',') if x]:active_counts[c]+=1
                support[key]={'decisions':int(len(g)),'mode_counts':dict(modes),'candidate_source_counts':dict(sources),'learned_support_decisions':int((g.active_components.astype(str)!='').sum()),'learned_support_rate':float((g.active_components.astype(str)!='').mean()),'nonzero_combined_q_decisions':int((g.combined_q.astype(float).abs()>1e-12).sum()),'nonzero_combined_q_rate':float((g.combined_q.astype(float).abs()>1e-12).mean()),'mean_state_support_visits':float(g.state_support_visits.astype(float).mean()),'active_component_counts':dict(active_counts)}
            (out/'policy_support_summary.json').write_text(json.dumps(support,indent=2,sort_keys=True))
    return summary
