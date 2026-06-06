"""Successor Representation (Dayan 1993).

Between model-free and model-based RL: learn a matrix M where
M(s, s') = E[sum_{k≥0} γ^k 1(s_k = s') | s_0 = s].

Then any value function decomposes as V(s) = sum_{s'} M(s, s') R(s').

SR is updated by TD-style learning:
    M(s, :) <- M(s, :) + α [1_{s'} + γ M(s', :) - M(s, :)]

Advantage: when rewards change but transitions don't, SR transfers
instantly. Hippocampal place cells are hypothesized to encode SR.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SuccessorRepresentation:
    n_states: int
    alpha: float = 0.1
    gamma: float = 0.95
    M: np.ndarray = field(default=None, repr=False)
    R_hat: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.M is None:
            self.M = np.eye(self.n_states)
        if self.R_hat is None:
            self.R_hat = np.zeros(self.n_states)

    def update(self, s: int, s_next: int, r: float) -> float:
        """TD update of SR matrix and reward estimate."""
        one_hot = np.zeros(self.n_states); one_hot[s] = 1
        target = one_hot + self.gamma * self.M[s_next]
        delta = target - self.M[s]
        self.M[s] += self.alpha * delta
        # Also learn reward associated with state s_next.
        self.R_hat[s_next] += self.alpha * (r - self.R_hat[s_next])
        return float(np.linalg.norm(delta))

    def value(self, s: int) -> float:
        """V(s) = M[s] · R_hat."""
        return float(self.M[s] @ self.R_hat)

    def values(self) -> np.ndarray:
        return self.M @ self.R_hat
