"""Population-vector decoding (Georgopoulos 1986).

Many cortical neurons are tuned to a "preferred direction" θ_i: their
firing rate is approximately

    r_i = b_i + g_i · cos(θ - θ_i)

for a stimulus direction θ. The population vector

    PV(θ) = sum_i (r_i - b_i) · u_i

where u_i = (cos θ_i, sin θ_i), points in the direction of the
stimulus. This is the dominant model of how motor cortex encodes
reaching directions.

Maximum-likelihood decoding given Poisson firing also gives back θ.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TunedPopulation:
    """N neurons with cosine tuning curves on the circle."""
    n: int
    preferred: np.ndarray = field(default=None, repr=False)
    baseline: float = 1.0
    gain: float = 5.0

    def __post_init__(self) -> None:
        if self.preferred is None:
            self.preferred = np.linspace(0, 2 * np.pi, self.n, endpoint=False)

    def firing_rates(self, theta: float) -> np.ndarray:
        return self.baseline + self.gain * np.maximum(
            np.cos(theta - self.preferred), 0.0)

    def sample(self, theta: float, rng: np.random.Generator | None = None,
                dt: float = 1.0) -> np.ndarray:
        rng = rng or np.random.default_rng(0)
        return rng.poisson(self.firing_rates(theta) * dt)

    def population_vector(self, counts: np.ndarray) -> tuple[float, float]:
        """Returns (theta_decoded, magnitude)."""
        c = np.maximum(counts - self.baseline, 0.0)
        x = (c * np.cos(self.preferred)).sum()
        y = (c * np.sin(self.preferred)).sum()
        return float(np.arctan2(y, x) % (2 * np.pi)), float(np.hypot(x, y))

    def maximum_likelihood(self, counts: np.ndarray,
                            n_grid: int = 360) -> float:
        """Brute-force MLE assuming Poisson firing rates."""
        thetas = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
        best_ll = -np.inf
        best_th = 0.0
        for th in thetas:
            r = self.firing_rates(th) + 1e-9
            ll = (counts * np.log(r) - r).sum()
            if ll > best_ll:
                best_ll = ll; best_th = th
        return float(best_th)


def angular_error(theta_hat: float, theta_true: float) -> float:
    """Smallest angular distance between two angles, in radians."""
    d = (theta_hat - theta_true) % (2 * np.pi)
    if d > np.pi:
        d -= 2 * np.pi
    return abs(d)
