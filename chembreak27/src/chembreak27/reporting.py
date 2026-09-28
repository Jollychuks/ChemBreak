from __future__ import annotations
import time

class Reporter:
    def __init__(self,total,enabled=True):
        self.total=total; self.enabled=bool(enabled); self.start=time.perf_counter(); self.completed=0; self.target_queries=0
    def start_task(self,phase,epoch,index,total,aid):
        if self.enabled:print(f'[{phase.upper()} E{epoch}] starting {index}/{total} | assignment={aid}',flush=True)
    def turn(self,phase,epoch,index,total,aid,turn,action,judge,reward,decision,source,policy_blocks=0):
        if not self.enabled:return
        print(
            f'[{phase.upper()} E{epoch}] {index}/{total} | assignment={aid} | turn={turn} | action={action} '
            f'| CHCS={int(judge.get("chcs",1))} | class={judge.get("response_class")} '
            f'| success={"YES" if judge.get("final_success") else "NO"} | reward={reward:+.3f} '
            f'| mode={decision.get("mode")} | source={source} '
            f'| eps={decision.get("base_epsilon",0):.2f}->{decision.get("effective_epsilon",0):.2f} '
            f'| Q={decision.get("combined_q",0):+.3f} | route_bonus={decision.get("route_bonus",0):+.3f} '
            f'| blocks={policy_blocks}',flush=True)
    def provider_block(self,phase,epoch,index,total,aid,action,code,n,limit):
        if self.enabled:print(f'[{phase.upper()} E{epoch}] {index}/{total} | assignment={aid} | ATTACK_LLM_POLICY_BLOCK | action={action} | code={code} | blocked_actions={n}/{limit} | target_queried=NO | continuing={"YES" if n<limit else "NO"}',flush=True)
    def complete(self,phase,epoch,index,total,aid,success,turns,reward):
        self.completed+=1
        if self.enabled:print(f'[{phase.upper()} E{epoch}] completed {index}/{total} | success={"YES" if success else "NO"} | turns={turns} | episode_reward={reward:+.3f} | target_queries={self.target_queries}',flush=True)
