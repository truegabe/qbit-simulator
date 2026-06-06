"""Oja's rule.

A Hebbian rule with normalization that extracts the first principal
component of the input data:

    Δw_i = η · y · (x_i - y · w_i)

After many presentations of zero-mean data, w converges to the
top eigenvector of the input covariance matrix, with ||w|| = 1.

Generalization to top-K PCs: Sanger's generalized Hebbian algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class OjaNeuron:
    n_inputs: int
    eta: float = 0.01
    w: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.w is None:
            rng = np.random.default_rng(0)
            self.w = rng.standard_normal(self.n_inputs)
            self.w /= np.linalg.norm(self.w) + 1e-12

    def step(self, x: np.ndarray) -> float:
        y = float(np.dot(self.w, x))
        self.w += self.eta * y * (x - y * self.w)
        return y

    def train(self, X: np.ndarray, n_epochs: int = 50,
               rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng(0)
        for _ in range(n_epochs):
            perm = rng.permutation(X.shape[0])
            for i in perm:
                self.step(X[i])
        return self.w


@dataclass
class SangerNetwork:
    """Generalized Hebbian: extract top-K principal components."""
    n_inputs: int
    n_components: int
    eta: float = 0.01
    W: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.W is None:
            rng = np.random.default_rng(0)
            self.W = rng.standard_normal((self.n_components, self.n_inputs))
            self.W /= np.linalg.norm(self.W, axis=1, keepdims=True) + 1e-12

    def step(self, x: np.ndarray) -> np.ndarray:
        y = self.W @ x
        # Sanger: subtract y_i times sum of prior components (lower triangular).
        outer = np.outer(y, x)
        proj = np.tril(np.outer(y, y)) @ self.W
        self.W += self.eta * (outer - proj)
        return y

    def train(self, X: np.ndarray, n_epochs: int = 50,
               rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng(0)
        for _ in range(n_epochs):
            for i in rng.permutation(X.shape[0]):
                self.step(X[i])
        return self.W
