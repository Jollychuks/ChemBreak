from __future__ import annotations
from abc import ABC, abstractmethod


class TargetSession(ABC):
    def __init__(self):
        self.history: list[dict[str, str]] = []

    @abstractmethod
    def ask(self, query: str) -> str:
        ...

    def reset(self) -> None:
        self.history.clear()

    def close(self) -> None:
        pass
