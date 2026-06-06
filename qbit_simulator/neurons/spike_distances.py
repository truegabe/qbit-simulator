"""Spike-train distances: Victor-Purpura and van Rossum.

Quantify the dissimilarity of two spike trains.

Victor-Purpura (1996):
  Edit distance: minimum total cost of transforming train A into B
  using
    - inserting / deleting a spike (cost 1)
    - shifting a spike by Δt (cost q · |Δt|)
  q is the temporal-precision parameter (1/q ≈ temporal resolution).

van Rossum (2001):
  Convolve both trains with an exponential kernel, take L² distance:
    f_A(t) = sum_i exp(-(t-t_i)/tau) Θ(t-t_i)
    D²(A, B) = (1/tau) ∫ (f_A - f_B)² dt
"""

from __future__ import annotations

import numpy as np


def victor_purpura(a: np.ndarray, b: np.ndarray, q: float = 1.0) -> float:
    """Spike-train edit distance.

    a, b: arrays of spike times.
    q   : cost per unit time shift.
    """
    n = len(a); m = len(b)
    if n == 0:
        return float(m)
    if m == 0:
        return float(n)
    D = np.zeros((n + 1, m + 1))
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(
                D[i - 1, j] + 1,
                D[i, j - 1] + 1,
                D[i - 1, j - 1] + q * abs(a[i - 1] - b[j - 1]),
            )
    return float(D[n, m])


def van_rossum(a: np.ndarray, b: np.ndarray, tau: float = 10.0,
               t_max: float | None = None, dt: float = 0.5) -> float:
    """van Rossum L² distance after exponential filtering."""
    if t_max is None:
        all_times = np.concatenate([a, b]) if len(a) + len(b) > 0 else np.array([0])
        t_max = float(all_times.max()) + 5 * tau if len(all_times) else 1.0
    t = np.arange(0.0, t_max + dt, dt)

    def filtered(spikes: np.ndarray) -> np.ndarray:
        f = np.zeros_like(t)
        for s in spikes:
            mask = t >= s
            f[mask] += np.exp(-(t[mask] - s) / tau)
        return f

    fa = filtered(a); fb = filtered(b)
    d2 = ((fa - fb) ** 2).sum() * dt / tau
    return float(np.sqrt(d2))


def spike_times_from_train(spike_train: np.ndarray, dt: float = 1.0
                            ) -> np.ndarray:
    """Convert boolean spike train (T,) into spike times array."""
    return np.where(spike_train)[0].astype(np.float64) * dt
