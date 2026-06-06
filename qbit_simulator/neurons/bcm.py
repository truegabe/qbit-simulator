"""BCM (Bienenstock-Cooper-Munro) learning rule.

A sliding-threshold Hebbian rule that produces selectivity:
synapses grow when post-synaptic activity exceeds a threshold θ, and
shrink when below it. The threshold itself slides with the time-
averaged squared post-activity:

    dw_i/dt = η · x_i · y · (y - θ)
    θ        = E[y^2]

A neuron with BCM weights learns to respond to a single (best) input
pattern in a set of orthogonal patterns — the canonical demonstration
of activity-dependent receptive-field formation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BCMNeuron:
    n_inputs: int
    eta: float = 0.005
    tau_theta: float = 50.0
    w: np.ndarray = field(default=None, repr=False)
    theta: float = 1.0

    def __post_init__(self) -> None:
        if self.w is None:
            self.w = np.random.default_rng(0).uniform(0.0, 0.1, self.n_inputs)

    def response(self, x: np.ndarray) -> float:
        return float(np.dot(self.w, x))

    def step(self, x: np.ndarray) -> float:
        y = self.response(x)
        # Synaptic update.
        self.w += self.eta * y * (y - self.theta) * x
        self.w = np.clip(self.w, 0.0, None)
        # Update sliding threshold (low-pass of y^2).
        self.theta += (y * y - self.theta) / self.tau_theta
        return y

    def train(self, X: np.ndarray, n_iter: int = 5000,
              rng: np.random.Generator | None = None) -> dict:
        """X: shape (n_patterns, n_inputs). Returns selectivity stats."""
        rng = rng or np.random.default_rng(0)
        for _ in range(n_iter):
            x = X[rng.integers(0, X.shape[0])]
            self.step(x)
        ys = np.array([self.response(x) for x in X])
        return {"weights": self.w.copy(), "theta": self.theta,
                "responses": ys, "selectivity_idx": int(np.argmax(ys))}
