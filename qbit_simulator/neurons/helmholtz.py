"""Helmholtz machine (Dayan, Hinton, Neal, Zemel 1995).

The original generative model trained by wake-sleep:
  - "Recognition" net q(h | x): bottom-up inference
  - "Generative" net p(x | h): top-down generation

Wake phase: given real x, sample h from q, train p to maximize p(x | h).
Sleep phase: given h ~ p(h), sample x from p(x | h), train q to recover h.

Both directions are layered sigmoid belief nets — binary neurons with
sigmoid activation, trained with delta-rule weight updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


@dataclass
class HelmholtzMachine:
    """Two-layer Helmholtz machine: visible + one hidden."""
    n_visible: int
    n_hidden: int
    eta: float = 0.05
    W_rec: np.ndarray = field(default=None, repr=False)   # x -> h
    W_gen: np.ndarray = field(default=None, repr=False)   # h -> x
    b_h:   np.ndarray = field(default=None, repr=False)
    b_v:   np.ndarray = field(default=None, repr=False)
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    def __post_init__(self) -> None:
        std = 0.1
        if self.W_rec is None:
            self.W_rec = self.rng.normal(0, std, (self.n_hidden, self.n_visible))
        if self.W_gen is None:
            self.W_gen = self.rng.normal(0, std, (self.n_visible, self.n_hidden))
        if self.b_h is None:
            self.b_h = np.zeros(self.n_hidden)
        if self.b_v is None:
            self.b_v = np.zeros(self.n_visible)

    def recognize(self, x: np.ndarray) -> np.ndarray:
        return sigmoid(self.W_rec @ x + self.b_h)

    def generate_visible(self, h: np.ndarray) -> np.ndarray:
        return sigmoid(self.W_gen @ h + self.b_v)

    def sample_prior_hidden(self) -> np.ndarray:
        # Uniform prior; can be replaced with a learned bias.
        return (self.rng.uniform(size=self.n_hidden) < 0.5).astype(np.float64)

    def wake(self, x: np.ndarray) -> None:
        """Train generative weights to reconstruct x."""
        q = self.recognize(x)
        h = (self.rng.uniform(size=self.n_hidden) < q).astype(np.float64)
        x_hat = self.generate_visible(h)
        err = x - x_hat
        self.W_gen += self.eta * np.outer(err, h)
        self.b_v   += self.eta * err

    def sleep(self) -> None:
        """Train recognition weights to invert generative model."""
        h = self.sample_prior_hidden()
        x = (self.rng.uniform(size=self.n_visible)
             < self.generate_visible(h)).astype(np.float64)
        q = self.recognize(x)
        err = h - q
        self.W_rec += self.eta * np.outer(err, x)
        self.b_h   += self.eta * err

    def train(self, X: np.ndarray, n_iter: int = 5000) -> None:
        for it in range(n_iter):
            x = X[self.rng.integers(0, X.shape[0])]
            self.wake(x)
            self.sleep()

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        h = self.recognize(x)
        return self.generate_visible(h)
