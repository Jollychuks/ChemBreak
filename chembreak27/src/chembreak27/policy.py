from __future__ import annotations
import json, random
from pathlib import Path
from typing import Any
from .constants import ACTIONS
from .utils import stable_hex, utc_now, write_json


class QPolicy:
    """Strict task-local Q policy for ChemBreak27.

    One QPolicy instance belongs to exactly one target-task pair. No Q-values or
    visit counts are shared across assignment IDs. The runner constructs a fresh
    policy at each task boundary and carries it only across that task's three
    adaptive epochs and frozen terminal evaluation.
    """
    SCHEMA_VERSION = 2

    def __init__(self, settings: dict[str, Any], seed: int, data: dict | None = None):
        self.settings = settings
        self.seed = int(seed)
        data = data or {}
        schema = int(data.get('policy_schema_version', self.SCHEMA_VERSION if not data else 0))
        if data and schema != self.SCHEMA_VERSION:
            raise RuntimeError(
                f'CB27 policy schema mismatch: expected {self.SCHEMA_VERSION}, found {schema}. '
                'Use fresh CB27 storage.'
            )
        self.q = dict(data.get('q', {}))
        self.visits = dict(data.get('visits', {}))
        self.stagnation = dict(data.get('stagnation', {}))
        self.decisions = int(data.get('decisions', 0))
        self.updates = int(data.get('updates', 0))
        self.frozen = bool(data.get('frozen', False))
        self.metadata = dict(data.get('metadata', {}))
        self.assignment_id = str(data.get('assignment_id', ''))

    @classmethod
    def load(cls, path, settings, seed):
        p = Path(path)
        return cls(settings, seed, json.loads(p.read_text()) if p.exists() else None)

    @staticmethod
    def _get(table, key, action, default=0.0):
        return float(table.get(str(key), {}).get(action, default))

    @staticmethod
    def _getv(table, key, action):
        return int(table.get(str(key), {}).get(action, 0))

    @staticmethod
    def _set(table, key, action, value):
        table.setdefault(str(key), {})[action] = value

    def bind_task(self, task_id: str):
        task_id = str(task_id)
        if self.assignment_id and self.assignment_id != task_id:
            raise RuntimeError(
                f'CB27 task-isolation violation: policy bound to {self.assignment_id}, '
                f'cannot bind to {task_id}'
            )
        self.assignment_id = task_id

    def _assert_task(self, task_id: str):
        task_id = str(task_id)
        if not self.assignment_id:
            self.assignment_id = task_id
        if self.assignment_id != task_id:
            raise RuntimeError(
                f'CB27 task-isolation violation: policy for {self.assignment_id} used by {task_id}'
            )

    def combined(self, task_id, keys, action):
        self._assert_task(task_id)
        key = keys['task']
        q = self._get(self.q, key, action)
        v = self._getv(self.visits, key, action)
        # Keep the reporting interface explicit while exposing that the global
        # component is absent in the task-local controller.
        return q, {'global': 0.0, 'task': q}, {'global': 0, 'task': v}, (['task'] if v > 0 else [])

    def _total_visits(self, task_id, keys, action):
        self._assert_task(task_id)
        return self._getv(self.visits, keys['task'], action)

    @staticmethod
    def _trailing_nonpositive(recent, action):
        n = 0
        for x in reversed(recent or []):
            if str(x.get('action')) != action or float(x.get('reward', 0)) > 0 or bool(x.get('success', False)):
                break
            n += 1
        return n

    def _state_support(self, task_id, keys, actions):
        return sum(self._total_visits(task_id, keys, a) for a in actions)

    def _effective_epsilon(self, task_id, keys, base, actions, recent):
        if self.frozen:
            return 0.0
        e = float(base)
        if self._state_support(task_id, keys, actions) == 0:
            e += float(self.settings.get('novel_state_epsilon_bonus', 0.0))
        if recent and float(recent[-1].get('reward', 0)) <= 0:
            e += float(self.settings.get('negative_feedback_epsilon_bonus', 0.0))
        return min(float(self.settings.get('max_effective_epsilon', .35)), max(0.0, e))

    def select(self, task_id, keys, epsilon, actions=None, recent=None, route_bonus=None, extra_blocked=None):
        self._assert_task(task_id)
        actions = list(actions or ACTIONS)
        recent = list(recent or [])
        route_bonus = dict(route_bonus or {})
        blocked = set(extra_blocked or [])
        hard = max(1, int(self.settings.get('hard_block_after_nonpositive_repeats', 2)))

        provider_valid = [a for a in actions if a not in blocked]
        state_stagnation = self.stagnation.get(str(keys['task']), {})
        stagnant = [a for a in provider_valid if int(state_stagnation.get(a, 0)) >= hard]
        # Stagnation blocks are state-specific and soft. Provider blocks remain hard.
        if len(provider_valid) > len(stagnant):
            blocked.update(stagnant)

        candidates = [a for a in actions if a not in blocked]
        if not candidates:
            return {
                'action': None, 'mode': 'no_available_action',
                'base_epsilon': float(epsilon), 'effective_epsilon': 0.0,
                'blocked_actions': sorted(blocked), 'combined_q': 0.0,
                'q_global': 0.0, 'q_task': 0.0, 'route_bonus': 0.0,
                'repeat_penalty': 0.0, 'state_support_visits': 0,
                'active_components': [], 'global_key': '', 'task_key': keys['task'],
            }

        idx = self.decisions + 1 if not self.frozen else 0
        if not self.frozen:
            self.decisions = idx
        eff = self._effective_epsilon(task_id, keys, epsilon, actions, recent)
        details = {a: self.combined(task_id, keys, a) for a in candidates}
        state_stagnation = self.stagnation.get(str(keys['task']), {})
        penalty = {
            a: float(self.settings.get('repeat_nonpositive_penalty', .75)) * int(state_stagnation.get(a, 0))
            for a in candidates
        }
        adjusted = {a: details[a][0] + float(route_bonus.get(a, 0.0)) - penalty[a] for a in candidates}
        supported = [a for a in candidates if details[a][3] or float(route_bonus.get(a, 0.0)) > 0]
        rng = random.Random(self.seed + idx * 7919)
        explore = (not self.frozen) and rng.random() < eff

        if explore:
            mv = min(self._total_visits(task_id, keys, a) for a in candidates)
            pool = [a for a in candidates if self._total_visits(task_id, keys, a) == mv]
            action = min(pool, key=lambda a: stable_hex(self.seed, idx, task_id, keys['task'], a))
            mode = 'exploration'
        elif not supported:
            mv = min(self._total_visits(task_id, keys, a) for a in candidates)
            pool = [a for a in candidates if self._total_visits(task_id, keys, a) == mv]
            action = min(pool, key=lambda a: stable_hex(self.seed, task_id, keys['task'], a))
            mode = 'cold_start'
        else:
            best = max(adjusted[a] for a in supported)
            pool = [a for a in supported if abs(adjusted[a] - best) < 1e-12]
            action = min(pool, key=lambda a: (self._total_visits(task_id, keys, a), stable_hex(self.seed, task_id, keys['task'], a)))
            mode = 'exploitation'

        q, vals, visits, active = details[action]
        return {
            'action': action, 'mode': mode,
            'base_epsilon': float(epsilon), 'effective_epsilon': float(eff),
            'blocked_actions': sorted(blocked), 'combined_q': float(q),
            'q_global': 0.0, 'q_task': float(vals['task']),
            'route_bonus': float(route_bonus.get(action, 0.0)),
            'repeat_penalty': float(penalty[action]),
            'state_support_visits': int(visits['task']),
            'active_components': active, 'global_key': '', 'task_key': keys['task'],
        }

    def update(self, task_id, keys, action, reward, next_keys, terminal):
        self._assert_task(task_id)
        if self.frozen:
            raise RuntimeError('Frozen CB27 task-local policy cannot be updated')
        key = keys['task']
        next_key = next_keys['task']
        old = self._get(self.q, key, action)
        nxt = 0.0 if terminal else max(self._get(self.q, next_key, a) for a in ACTIONS)
        target = float(reward) if terminal else float(reward) + float(self.settings['discount']) * nxt
        alpha = float(self.settings.get('learning_rate', self.settings.get('task_learning_rate', .18)))
        new = old + alpha * (target - old)
        self._set(self.q, key, action, new)
        self._set(self.visits, key, action, self._getv(self.visits, key, action) + 1)
        sc = self.stagnation.setdefault(str(key), {})
        sc[action] = 0 if float(reward) > 0 or bool(terminal) else int(sc.get(action, 0)) + 1
        self.updates += 1
        return {
            'old_q_global': 0.0, 'new_q_global': 0.0,
            'old_q_task': old, 'new_q_task': new,
        }

    def to_dict(self):
        return {
            'namespace': 'CB27', 'policy_schema_version': self.SCHEMA_VERSION,
            'seed': self.seed, 'assignment_id': self.assignment_id,
            'metadata': self.metadata, 'q': self.q, 'visits': self.visits, 'stagnation': self.stagnation,
            'decisions': self.decisions, 'updates': self.updates,
            'frozen': self.frozen, 'saved_at_utc': utc_now(),
        }

    def save(self, path):
        write_json(path, self.to_dict())

    def freeze(self, path):
        self.frozen = True
        self.save(path)
        return self.to_dict()

    def summary(self):
        return {
            'task_states': len(self.q), 'decisions': self.decisions,
            'updates': self.updates, 'frozen': self.frozen,
            'assignment_id': self.assignment_id,
            'cross_task_transfer': False,
        }
