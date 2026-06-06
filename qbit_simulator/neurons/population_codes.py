"""Probabilistic Population Codes (Ma, Beck, Latham, Pouget 2006).

A population of Poisson-firing neurons with tuning curves f_i(s)
naturally encodes a probability distribution over the stimulus:

    p(s | r) ∝ exp(sum_i r_i log f_i(s) - sum_i f_i(s))

Two populations with PPC representations can be multiplicatively
combined by simple ADDITION of their spike counts — neurally
plausible Bayes-optimal cue integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ProbabilisticPopulationCode:
    n_neurons: int
    s_grid: np.ndarray = field(default=None, repr=False)
    preferred: np.ndarray = field(default=None, repr=False)
    width: float = 0.5
    gain: float = 5.0

    def __post_init__(self) -> None:
        if self.s_grid is None:
            self.s_grid = np.linspace(-5, 5, 200)
        if self.preferred is None:
            self.preferred = np.linspace(-3, 3, self.n_neurons)

    def tuning(self, s: float | np.ndarray) -> np.ndarray:
        """Gaussian tuning curves."""
        s_arr = np.atleast_1d(s)
        return self.gain * np.exp(-(s_arr[:, None] - self.preferred[None, :]) ** 2
                                    / (2 * self.width ** 2))

    def posterior(self, counts: np.ndarray) -> np.ndarray:
        """Compute p(s | counts) on s_grid."""
        f = self.tuning(self.s_grid)             # (n_s, n_neurons)
        log_p = (counts[None, :] * np.log(f + 1e-9) - f).sum(axis=1)
        log_p -= log_p.max()
        p = np.exp(log_p)
        return p / p.sum()

    def estimate(self, counts: np.ndarray) -> float:
        """MAP estimate from posterior."""
        p = self.posterior(counts)
        return float(self.s_grid[np.argmax(p)])

    @staticmethod
    def fuse(counts_a: np.ndarray, counts_b: np.ndarray) -> np.ndarray:
        """Bayes-optimal cue integration = sum of counts."""
        return counts_a + counts_b


def sample_counts(ppc: ProbabilisticPopulationCode, s: float,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    rates = ppc.gain * np.exp(-(s - ppc.preferred) ** 2 / (2 * ppc.width ** 2))
    return rng.poisson(rates)
