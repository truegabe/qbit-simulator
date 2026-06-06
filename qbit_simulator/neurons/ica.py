"""Independent Component Analysis (FastICA, Hyvärinen 1999).

Find a linear unmixing matrix W such that y = W x has independent
components. Uses the fixed-point iteration:

    w <- E[x g(w^T x)] - E[g'(w^T x)] w
    w <- w / ||w||

with non-linearity g(u) = tanh(u). After K components are found, each
new w is orthogonalized against previously-found ones (deflation).

For zero-mean whitened input, ICA recovers independent latent sources
(blind source separation).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def whiten(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center + whiten data X (samples in rows)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = Xc.T @ Xc / Xc.shape[0]
    U, s, _ = np.linalg.svd(cov)
    W_white = U @ np.diag(1.0 / np.sqrt(s + 1e-9)) @ U.T
    return Xc @ W_white, W_white


@dataclass
class FastICA:
    n_components: int
    max_iter: int = 200
    tol: float = 1e-5
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)
    W: np.ndarray = field(default=None, repr=False)
    W_white: np.ndarray = field(default=None, repr=False)
    mean: np.ndarray = field(default=None, repr=False)

    def fit(self, X: np.ndarray) -> None:
        self.mean = X.mean(axis=0)
        X_white, self.W_white = whiten(X)
        n, d = X_white.shape
        W = self.rng.normal(0, 1, (self.n_components, d))
        for i in range(self.n_components):
            w = W[i]; w /= np.linalg.norm(w) + 1e-9
            for _ in range(self.max_iter):
                u = X_white @ w
                g  = np.tanh(u)
                g_prime = 1 - g ** 2
                w_new = (X_white.T @ g) / n - g_prime.mean() * w
                # Deflate: subtract projections onto previous rows.
                for j in range(i):
                    w_new -= (w_new @ W[j]) * W[j]
                w_new /= np.linalg.norm(w_new) + 1e-9
                if np.abs(np.abs(w @ w_new) - 1) < self.tol:
                    w = w_new
                    break
                w = w_new
            W[i] = w
        self.W = W

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xc = X - self.mean
        Xw = Xc @ self.W_white
        return Xw @ self.W.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
