"""Restricted Boltzmann Machine (RBM) — classical training.

A two-layer energy-based model: visible v and hidden h binary units
fully connected (within layers: not connected).

    E(v, h) = -v^T W h - b^T v - c^T h
    p(v, h) ∝ exp(-E(v, h))

Trained by Contrastive Divergence (CD-k):
    Δ W = η (E_data[v h^T] - E_model[v h^T])

The model can also be unrolled into a DBN by stacking RBMs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


@dataclass
class RBM:
    """Restricted Boltzmann Machine with binary units."""
    n_visible: int
    n_hidden: int
    eta: float = 0.05
    W: np.ndarray = field(default=None, repr=False)
    b: np.ndarray = field(default=None, repr=False)
    c: np.ndarray = field(default=None, repr=False)
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    def __post_init__(self) -> None:
        if self.W is None:
            self.W = self.rng.normal(0, 0.1, (self.n_visible, self.n_hidden))
        if self.b is None:
            self.b = np.zeros(self.n_visible)
        if self.c is None:
            self.c = np.zeros(self.n_hidden)

    def p_h_given_v(self, v: np.ndarray) -> np.ndarray:
        return sigmoid(v @ self.W + self.c)

    def p_v_given_h(self, h: np.ndarray) -> np.ndarray:
        return sigmoid(h @ self.W.T + self.b)

    def sample(self, p: np.ndarray) -> np.ndarray:
        return (self.rng.uniform(size=p.shape) < p).astype(np.float64)

    def cd_k(self, v0: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Contrastive divergence k steps. Returns (v_neg, h_neg)."""
        h = self.sample(self.p_h_given_v(v0))
        v = v0
        for _ in range(k):
            v = self.sample(self.p_v_given_h(h))
            h = self.sample(self.p_h_given_v(v))
        return v, h

    def step(self, v0: np.ndarray, k: int = 1) -> float:
        """Single CD-k update. Returns reconstruction error."""
        p_h0 = self.p_h_given_v(v0)
        v_neg, _ = self.cd_k(v0, k=k)
        p_h_neg = self.p_h_given_v(v_neg)
        # Per-sample stats.
        self.W += self.eta * (np.outer(v0, p_h0) - np.outer(v_neg, p_h_neg))
        self.b += self.eta * (v0 - v_neg)
        self.c += self.eta * (p_h0 - p_h_neg)
        return float(((v0 - v_neg) ** 2).sum())

    def train(self, X: np.ndarray, n_epochs: int = 50, k: int = 1) -> list:
        losses = []
        for ep in range(n_epochs):
            ep_loss = 0.0
            for i in self.rng.permutation(X.shape[0]):
                ep_loss += self.step(X[i], k=k)
            losses.append(ep_loss / X.shape[0])
        return losses

    def reconstruct(self, v: np.ndarray) -> np.ndarray:
        h = self.p_h_given_v(v)
        return self.p_v_given_h(h)
