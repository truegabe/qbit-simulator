"""Cerebellum model (Marr-Albus, supervised motor learning).

Architecture:
  - Mossy fibers carry context/state info to many granule cells.
  - Granule cells form a sparse expansion code (high-D random projection).
  - Purkinje cells are output: their weights from granule cells are
    plastic and trained by climbing fibers (error signals from inferior
    olive).
  - Climbing-fiber-induced LTD reduces weights of recently-active
    parallel fibers when an error occurs.

This is essentially a perceptron with a sparse random expansion — a
classical model of supervised motor learning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CerebellumModel:
    n_mossy: int
    n_granule: int = 500
    n_purkinje: int = 1
    sparsity: float = 0.05   # fraction of granule cells active per pattern
    eta: float = 0.05
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)
    W_mg: np.ndarray = field(default=None, repr=False)
    W_gp: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.W_mg is None:
            # Sparse random projection mossy → granule.
            self.W_mg = self.rng.normal(
                0.0, 1.0 / np.sqrt(self.n_mossy),
                size=(self.n_granule, self.n_mossy))
        if self.W_gp is None:
            self.W_gp = np.zeros((self.n_purkinje, self.n_granule))

    def granule_response(self, x: np.ndarray) -> np.ndarray:
        """Sparse non-linear expansion via top-K activation."""
        h = self.W_mg @ x
        k = max(int(self.sparsity * self.n_granule), 1)
        # Top-k binarization.
        thresh = np.partition(h, -k)[-k]
        return (h >= thresh).astype(np.float64)

    def predict(self, x: np.ndarray) -> np.ndarray:
        g = self.granule_response(x)
        return self.W_gp @ g

    def update(self, x: np.ndarray, target: np.ndarray) -> float:
        g = self.granule_response(x)
        y = self.W_gp @ g
        err = target - y
        # LTD: climbing fiber = err, parallel fiber = g.
        self.W_gp += self.eta * np.outer(err, g)
        return float(0.5 * (err ** 2).sum())

    def train(self, X: np.ndarray, Y: np.ndarray,
              n_epochs: int = 100) -> list:
        losses = []
        for ep in range(n_epochs):
            ep_loss = 0.0
            for i in self.rng.permutation(X.shape[0]):
                ep_loss += self.update(X[i], Y[i])
            losses.append(ep_loss / X.shape[0])
        return losses
