from __future__ import annotations
import time

class Reporter:
    def __init__(self,total=120,enabled=True):
        self.total=int(total); self.enabled=bool(enabled); self.started=time.time()
    def start_task(self,phase,epoch,index,phase_total,assignment_id):
        if not self.enabled:return
        ep=f' E{epoch}' if epoch else ''
        print(f'[{phase.upper()}{ep}] starting {index}/{phase_total} | assignment={assignment_id}',flush=True)
    def turn(self,phase,epoch,index,phase_total,assignment_id,turn_index,action_id,judge,reward,decision=None,candidate_source=''):
        if not self.enabled:return
        ep=f' E{epoch}' if epoch else ''; cls=str(judge.get('response_class','unknown')); success='YES' if bool(judge.get('success')) else 'NO'; prog=float(judge.get('goal_progress',0.0)); diag=''
        if decision:
            blocked=','.join(decision.get('blocked_actions',[])) or '-'; active=','.join(decision.get('active_components',[])) or '-'
            ev=''
            if decision.get('trajectory_rank_score') is not None:ev=f" | trajectory_rank={float(decision['trajectory_rank_score']):.3f}"
            elif decision.get('evidence_rank_score') is not None:ev=f" | evidence_rank={float(decision['evidence_rank_score']):.3f}"
            diag=(f" | mode={decision.get('mode','?')} | source={candidate_source or '-'} | eps={float(decision.get('base_epsilon',0)):.2f}->{float(decision.get('effective_epsilon',0)):.2f}"
                  f" | Qg={float(decision.get('q_global',0)):+.3f} | Qhc={float(decision.get('q_hc',0)):+.3f} | Qhd={float(decision.get('q_hd',0)):+.3f}"
                  f" | Qot={float(decision.get('q_ot',0)):+.3f} | Qt={float(decision.get('q_task',0)):+.3f} | Q={float(decision.get('combined_q',0)):+.3f}"
                  f" | active={active} | support={int(decision.get('state_support_visits',0))} | repeat_penalty={float(decision.get('repeat_penalty',0)):.3f}"
                  f" | score={float(decision.get('adjusted_score',0)):+.3f} | blocked={blocked}{ev}")
        print(f'[{phase.upper()}{ep}] {index}/{phase_total} | assignment={assignment_id} | turn={turn_index} | action={action_id} | class={cls} | success={success} | goal_progress={prog:.2f} | reward={float(reward):+.3f}{diag}',flush=True)
    def provider_block(self,phase,epoch,index,phase_total,assignment_id,action_id,error_code,blocked_count,max_blocks,continuing=True):
        if not self.enabled:return
        ep=f' E{epoch}' if epoch else ''
        print(f'[{phase.upper()}{ep}] {index}/{phase_total} | assignment={assignment_id} | ATTACK_LLM_POLICY_BLOCK | action={action_id} | code={error_code} | blocked_actions={blocked_count}/{max_blocks} | target_queried=NO | continuing={"YES" if continuing else "NO"}',flush=True)
    def episode(self,phase,epoch,index,phase_total,success,turns,*,phase_done,phase_successes,overall_completed,query_total,total_reward=None):
        if not self.enabled:return
        elapsed=max(time.time()-self.started,1e-6); rate=max(overall_completed,1)/elapsed; remaining=max(self.total-overall_completed,0)/rate if rate else 0; ep=f' E{epoch}' if epoch else ''; asr=phase_successes/phase_done if phase_done else 0
        rt='' if total_reward is None else f' | episode_reward={float(total_reward):+.3f}'
        print(f'[{phase.upper()}{ep}] completed {index}/{phase_total} | running_ASR={100*asr:.1f}% ({phase_successes}/{phase_done}) | success={"YES" if success else "NO"} | turns={turns}{rt} | overall={overall_completed}/{self.total} | target_queries={query_total} | elapsed={elapsed/60:.1f}m | ETA={remaining/60:.1f}m',flush=True)
