"""Mean-field inference on Boltzmann / Ising-like networks.

For a binary network with energy
    E(s) = -0.5 s^T J s - h^T s,
mean-field approximation: q(s) = prod_i Bernoulli(m_i), with the
self-consistent fixed point

    m_i = σ(h_i + sum_j J_ij m_j).

Iterate to convergence — this is the simplest variational inference
and the dominant computational hypothesis for cortical rate dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


@dataclass
class MeanFieldBoltzmann:
    n: int
    J: np.ndarray = field(default=None, repr=False)
    h: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.J is None:
            self.J = np.zeros((self.n, self.n))
        if self.h is None:
            self.h = np.zeros(self.n)

    def mean_field(self, n_iter: int = 100, tol: float = 1e-6,
                    damping: float = 0.5) -> np.ndarray:
        m = sigmoid(self.h)
        for it in range(n_iter):
            field = self.h + self.J @ m
            m_new = (1 - damping) * sigmoid(field) + damping * m
            if np.max(np.abs(m_new - m)) < tol:
                m = m_new
                break
            m = m_new
        return m

    def variational_free_energy(self, m: np.ndarray) -> float:
        """F(m) = -<E>_q - H(q) where q is mean-field Bernoulli(m)."""
        m_safe = np.clip(m, 1e-9, 1 - 1e-9)
        energy = -0.5 * m @ self.J @ m - self.h @ m
        entropy = -(m_safe * np.log(m_safe)
                     + (1 - m_safe) * np.log(1 - m_safe)).sum()
        return float(energy - entropy)


def gibbs_sample(mfb: MeanFieldBoltzmann, n_samples: int = 1000,
                  burn_in: int = 500,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Reference samples via Gibbs sampling for comparison with MF."""
    rng = rng or np.random.default_rng(0)
    s = (rng.uniform(size=mfb.n) > 0.5).astype(np.float64)
    out = np.zeros((n_samples, mfb.n))
    for t in range(burn_in + n_samples):
        for i in range(mfb.n):
            p = sigmoid(mfb.h[i] + mfb.J[i] @ s)
            s[i] = float(rng.uniform() < p)
        if t >= burn_in:
            out[t - burn_in] = s
    return out
