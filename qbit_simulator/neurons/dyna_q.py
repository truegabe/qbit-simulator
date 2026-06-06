"""Dyna-Q (Sutton 1991): model-based RL.

Standard Q-learning + a learned model of the environment used for
"planning" updates between real steps:

    1. Take action a in state s, observe r, s'.
    2. Q-learning update on Q(s, a).
    3. Add (s, a, r, s') to the model.
    4. For k planning steps:
         (s', a') ~ model.sample()
         Q-learning update on Q(s', a') from model prediction.

Speeds up learning enormously by reusing past experience.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DynaQ:
    n_states: int
    n_actions: int
    alpha: float = 0.1
    gamma: float = 0.95
    eps: float = 0.1
    n_plan: int = 5
    Q: np.ndarray = field(default=None, repr=False)
    model: dict = field(default_factory=dict)
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    def __post_init__(self) -> None:
        if self.Q is None:
            self.Q = np.zeros((self.n_states, self.n_actions))

    def act(self, s: int) -> int:
        if self.rng.uniform() < self.eps:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.Q[s]))

    def _q_update(self, s: int, a: int, r: float, s_next: int,
                   done: bool) -> None:
        target = r if done else r + self.gamma * np.max(self.Q[s_next])
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def step(self, s: int, a: int, r: float, s_next: int,
              done: bool = False) -> None:
        self._q_update(s, a, r, s_next, done)
        # Record in model.
        self.model[(s, a)] = (r, s_next, done)
        # Planning.
        keys = list(self.model.keys())
        for _ in range(min(self.n_plan, len(keys))):
            idx = self.rng.integers(len(keys))
            sk, ak = keys[idx]
            rk, snk, dk = self.model[(sk, ak)]
            self._q_update(sk, ak, rk, snk, dk)
