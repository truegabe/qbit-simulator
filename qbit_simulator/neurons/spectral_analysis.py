"""Spectral analysis of random recurrent networks.

For an N×N random matrix with i.i.d. entries of variance g²/N, the
eigenvalues fill a disk of radius g in the complex plane
(Girko's circular law).

This module provides:
  - Spectral radius computation.
  - Stability check for linear rate networks dx/dt = -x + W x.
  - Lyapunov exponent estimator.
"""

from __future__ import annotations

import numpy as np


def spectral_radius(W: np.ndarray) -> float:
    """Largest absolute eigenvalue."""
    return float(np.max(np.abs(np.linalg.eigvals(W))))


def is_stable(W: np.ndarray, tau: float = 1.0) -> bool:
    """For dx/dt = (-x + W x)/tau, stable iff Re(λ_W) < 1 ∀ λ."""
    evals = np.linalg.eigvals(W)
    return bool(np.all(evals.real < 1.0))


def lyapunov_exponent(W: np.ndarray, n_steps: int = 200,
                       activation=None) -> float:
    """Estimate largest Lyapunov exponent via two-trajectory perturbation."""
    n = W.shape[0]
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n) * 0.1
    eps = 1e-6
    y = x + eps * rng.standard_normal(n)
    delta = y - x
    delta /= np.linalg.norm(delta)
    log_growth = 0.0
    for t in range(n_steps):
        x_new = W @ (np.tanh(x) if activation is None else activation(x))
        y_new = W @ (np.tanh(y) if activation is None else activation(y))
        d = y_new - x_new
        norm = np.linalg.norm(d)
        if norm > 0:
            log_growth += np.log(norm / eps)
        d /= max(norm, 1e-12)
        x = x_new
        y = x + eps * d
    return float(log_growth / n_steps)


def random_recurrent_matrix(n: int, g: float = 1.0,
                              rng: np.random.Generator | None = None) -> np.ndarray:
    """N×N matrix with i.i.d. N(0, g²/N) entries. Spectral radius ≈ g."""
    rng = rng or np.random.default_rng(0)
    return rng.normal(0, g / np.sqrt(n), (n, n))
