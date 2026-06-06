"""Self-Organizing Map (Kohonen 1982).

Unsupervised learning of a topology-preserving low-dimensional map
from high-dimensional input. Each unit has a weight vector w_i in
input space. On each input x:
  1. Find best matching unit (BMU): i* = argmin_i ||x - w_i||.
  2. Update w_j of nearby units in MAP space:
       w_j <- w_j + alpha * h(j, i*) * (x - w_j)
     where h is a Gaussian neighborhood in map space.

Both alpha and the neighborhood width shrink over training.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SelfOrganizingMap:
    """2D SOM with rectangular grid."""
    map_h: int
    map_w: int
    input_dim: int
    alpha0: float = 0.5
    sigma0: float = None  # default to max(map_h, map_w)/2
    weights: np.ndarray = field(default=None, repr=False)
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    def __post_init__(self) -> None:
        if self.sigma0 is None:
            self.sigma0 = max(self.map_h, self.map_w) / 2.0
        if self.weights is None:
            self.weights = self.rng.uniform(
                size=(self.map_h, self.map_w, self.input_dim))

    def bmu(self, x: np.ndarray) -> tuple[int, int]:
        d = np.linalg.norm(self.weights - x[None, None, :], axis=-1)
        idx = np.argmin(d)
        return idx // self.map_w, idx % self.map_w

    def _neighborhood(self, bmu: tuple[int, int], sigma: float) -> np.ndarray:
        rows = np.arange(self.map_h)[:, None]
        cols = np.arange(self.map_w)[None, :]
        d2 = (rows - bmu[0]) ** 2 + (cols - bmu[1]) ** 2
        return np.exp(-d2 / (2 * sigma * sigma))

    def update(self, x: np.ndarray, alpha: float, sigma: float) -> None:
        i, j = self.bmu(x)
        h = self._neighborhood((i, j), sigma)
        delta = alpha * h[:, :, None] * (x[None, None, :] - self.weights)
        self.weights += delta

    def train(self, X: np.ndarray, n_iter: int = 1000) -> None:
        """Train on N input samples in X (shape (N, input_dim)).

        Standard exponential decay schedule for alpha and sigma.
        """
        tau = n_iter / np.log(self.sigma0)
        for t in range(n_iter):
            x = X[self.rng.integers(0, X.shape[0])]
            alpha = self.alpha0 * np.exp(-t / n_iter)
            sigma = max(self.sigma0 * np.exp(-t / tau), 0.5)
            self.update(x, alpha, sigma)

    def quantization_error(self, X: np.ndarray) -> float:
        errs = []
        for x in X:
            i, j = self.bmu(x)
            errs.append(np.linalg.norm(x - self.weights[i, j]))
        return float(np.mean(errs))
