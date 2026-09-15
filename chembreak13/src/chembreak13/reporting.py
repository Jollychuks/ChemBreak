from __future__ import annotations
import time
class Reporter:
    def __init__(self,total=120): self.total=total; self.started=time.time(); self.completed=0; self.queries=0
    def resume(self,completed,queries): self.completed=completed; self.queries=queries
    def episode(self,phase,epoch,index,phase_total,success,turns):
        self.completed+=1; self.queries+=turns; elapsed=max(time.time()-self.started,1e-6); rate=self.completed/elapsed; remaining=(self.total-self.completed)/rate if rate else 0
        ep=f" E{epoch}" if epoch else ''
        print(f"[{phase.upper()}{ep}] {index}/{phase_total} | overall {self.completed}/{self.total} ({100*self.completed/self.total:.1f}%) | success={'YES' if success else 'NO'} | turns={turns} | target_queries={self.queries} | elapsed={elapsed/60:.1f}m | ETA={remaining/60:.1f}m")
