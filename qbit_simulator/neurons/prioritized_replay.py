"""Prioritized memory replay (Schaul et al. 2016).

Instead of uniform random sampling of past experiences during replay,
prioritize transitions with high TD-error |δ| — those that taught the
agent the most. This produces much faster learning.

Sampling probability:
    P(i) ∝ |δ_i|^α

To correct for the resulting bias in expected-value updates, use
importance sampling weights:
    w_i = (1 / (N · P(i)))^β
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PrioritizedReplayBuffer:
    capacity: int = 1000
    alpha: float = 0.6
    beta: float = 0.4
    eps: float = 1e-3
    buffer: list = field(default_factory=list)
    priorities: list = field(default_factory=list)
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    def add(self, transition, td_error: float = 1.0) -> None:
        p = (abs(td_error) + self.eps) ** self.alpha
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(p)
        else:
            # Replace random old entry.
            idx = self.rng.integers(self.capacity)
            self.buffer[idx] = transition
            self.priorities[idx] = p

    def sample(self, batch_size: int) -> tuple[list, np.ndarray, np.ndarray]:
        if not self.buffer:
            return [], np.array([]), np.array([])
        priors = np.array(self.priorities)
        probs = priors / priors.sum()
        idx = self.rng.choice(len(self.buffer), size=min(batch_size, len(self.buffer)),
                                p=probs, replace=False)
        batch = [self.buffer[i] for i in idx]
        # Importance-sampling weights.
        n = len(self.buffer)
        weights = (n * probs[idx]) ** (-self.beta)
        weights /= weights.max() + 1e-12
        return batch, idx, weights

    def update_priorities(self, idx: np.ndarray, td_errors: np.ndarray) -> None:
        for i, e in zip(idx, td_errors):
            self.priorities[i] = (abs(e) + self.eps) ** self.alpha
