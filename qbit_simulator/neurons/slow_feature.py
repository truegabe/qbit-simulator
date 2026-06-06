"""Slow Feature Analysis (Wiskott & Sejnowski 2002).

Find features y_j = g_j(x) that vary as SLOWLY as possible over time
while remaining decorrelated and unit-variance:

    minimize  E[(dy_j/dt)^2]
    subject to E[y_j] = 0, E[y_j^2] = 1, E[y_i y_j] = 0 (i ≠ j)

Linear SFA: solution is the generalized eigenvalue problem
    Σ_{Δx} w = λ Σ_x w
where Σ_x is data covariance, Σ_{Δx} is the temporal-difference
covariance. Eigenvectors with SMALLEST eigenvalues are the slowest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SlowFeatureAnalysis:
    n_components: int
    W: np.ndarray = field(default=None, repr=False)
    mean: np.ndarray = field(default=None, repr=False)
    eigvals: np.ndarray = field(default=None, repr=False)

    def fit(self, X: np.ndarray) -> None:
        """X: shape (T, d). Time samples assumed sequential."""
        self.mean = X.mean(axis=0)
        Xc = X - self.mean
        T = Xc.shape[0]
        # Sphere the data.
        cov = Xc.T @ Xc / T
        # Generalized eigvals: solve cov_dx @ w = lambda * cov @ w.
        # Equivalent to standard eig on cov^{-1/2} cov_dx cov^{-1/2}.
        evals, evecs = np.linalg.eigh(cov)
        # Regularize tiny eigenvalues.
        inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(np.maximum(evals, 1e-9))) @ evecs.T
        Y = Xc @ inv_sqrt
        dY = np.diff(Y, axis=0)
        cov_dy = dY.T @ dY / (T - 1)
        ev2, vec2 = np.linalg.eigh(cov_dy)
        # Slowest features have SMALLEST eigenvalues.
        idx = np.argsort(ev2)[:self.n_components]
        self.W = inv_sqrt @ vec2[:, idx]
        self.eigvals = ev2[idx]

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) @ self.W

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
