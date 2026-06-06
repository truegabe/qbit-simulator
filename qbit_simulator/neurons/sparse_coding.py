"""Sparse coding (Olshausen & Field 1996).

Learn a dictionary D such that data X ≈ D s where s is sparse.

    min_D, s    0.5 ||x - D s||^2 + λ ||s||_1

Alternating minimization:
  - Given D fixed, find s by ISTA (Iterative Shrinkage-Thresholding):
        s <- soft_threshold(s + η D^T (x - D s), λ η)
  - Given s fixed, gradient on D:
        D <- D + α (x - D s) s^T
  - Normalize columns of D.

Trained on natural-image patches, the dictionary atoms become
oriented edge filters — qualitatively matching V1 simple cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def soft_threshold(x: np.ndarray, lam: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)


@dataclass
class SparseCoder:
    n_features: int
    n_atoms: int = 64
    lam: float = 0.1
    eta_inf: float = 0.1
    eta_dict: float = 0.005
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)
    D: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.D is None:
            self.D = self.rng.normal(0, 1, (self.n_features, self.n_atoms))
            self.D /= np.linalg.norm(self.D, axis=0, keepdims=True) + 1e-9

    def infer(self, x: np.ndarray, n_iter: int = 100) -> np.ndarray:
        s = np.zeros(self.n_atoms)
        for _ in range(n_iter):
            grad = self.D.T @ (self.D @ s - x)
            s = soft_threshold(s - self.eta_inf * grad, self.lam * self.eta_inf)
        return s

    def update(self, x: np.ndarray) -> float:
        s = self.infer(x)
        residual = x - self.D @ s
        # Gradient on D.
        self.D += self.eta_dict * np.outer(residual, s)
        # Normalize columns.
        self.D /= np.linalg.norm(self.D, axis=0, keepdims=True) + 1e-9
        return float(0.5 * (residual @ residual) + self.lam * np.abs(s).sum())

    def train(self, X: np.ndarray, n_iter: int = 1000) -> list:
        losses = []
        for it in range(n_iter):
            x = X[self.rng.integers(X.shape[0])]
            losses.append(self.update(x))
        return losses
