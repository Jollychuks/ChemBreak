from __future__ import annotations
import time

class Reporter:
    """Notebook-friendly live progress reporter.

    It intentionally prints compact metrics rather than prompt/response text.
    Every print uses flush=True so Notebook Enterprise displays progress as the
    run advances instead of waiting for the phase cell to finish.
    """
    def __init__(self,total=120,enabled=True):
        self.total=int(total)
        self.enabled=bool(enabled)
        self.started=time.time()

    def start_task(self,phase,epoch,index,phase_total,assignment_id):
        if not self.enabled: return
        ep=f" E{epoch}" if epoch else ''
        print(f"[{phase.upper()}{ep}] starting {index}/{phase_total} | assignment={assignment_id}",flush=True)

    def turn(self,phase,epoch,index,phase_total,assignment_id,turn_index,action_id,judge,reward):
        if not self.enabled: return
        ep=f" E{epoch}" if epoch else ''
        cls=str(judge.get('response_class','unknown'))
        success='YES' if bool(judge.get('success')) else 'NO'
        progress=float(judge.get('goal_progress',0.0))
        print(
            f"[{phase.upper()}{ep}] {index}/{phase_total} | assignment={assignment_id} | "
            f"turn={turn_index} | action={action_id} | class={cls} | "
            f"success={success} | goal_progress={progress:.2f} | reward={float(reward):+.3f}",
            flush=True,
        )

    def episode(self,phase,epoch,index,phase_total,success,turns,*,phase_done,phase_successes,overall_completed,query_total,total_reward=None):
        if not self.enabled: return
        elapsed=max(time.time()-self.started,1e-6)
        rate=max(overall_completed,1)/elapsed
        remaining=max(self.total-overall_completed,0)/rate if rate else 0
        ep=f" E{epoch}" if epoch else ''
        asr=(phase_successes/phase_done) if phase_done else 0.0
        reward_text='' if total_reward is None else f" | episode_reward={float(total_reward):+.3f}"
        print(
            f"[{phase.upper()}{ep}] completed {index}/{phase_total} | "
            f"running_ASR={100*asr:.1f}% ({phase_successes}/{phase_done}) | "
            f"success={'YES' if success else 'NO'} | turns={turns}{reward_text} | "
            f"overall={overall_completed}/{self.total} | target_queries={query_total} | "
            f"elapsed={elapsed/60:.1f}m | ETA={remaining/60:.1f}m",
            flush=True,
        )
