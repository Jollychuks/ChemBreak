from __future__ import annotations
import time


class Progress:
    def __init__(self, label: str, total: int):
        self.label = label
        self.total = max(total, 1)
        self.start = time.time()
        self.done = 0

    def _print(self, note: str = "") -> None:
        elapsed = time.time() - self.start
        rate = self.done / elapsed if elapsed else 0
        eta = (self.total - self.done) / rate if rate else 0
        pct = 100 * self.done / self.total
        print(f"[{self.label}] {self.done}/{self.total} ({pct:5.1f}%) | elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m | {note}", flush=True)

    def step(self, note: str = "") -> None:
        self.done = min(self.total, self.done + 1)
        self._print(note)

    def advance(self, count: int, note: str = "") -> None:
        self.done = min(self.total, self.done + max(0, int(count)))
        self._print(note)
