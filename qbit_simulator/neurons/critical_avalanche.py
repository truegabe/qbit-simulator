"""Critical-state neuronal avalanches (Beggs & Plenz 2003).

Cortical networks operating near a critical point exhibit
"avalanches" — cascades of activity whose sizes S and durations D
follow power laws:
    P(S) ~ S^{-3/2}
    P(D) ~ D^{-2}

This module simulates a branching-process model where each spiking
neuron triggers, on average, σ post-synaptic spikes. σ = 1 is the
critical regime.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BranchingNetwork:
    n: int = 200
    sigma: float = 1.0
    p_ext: float = 0.001
    state: np.ndarray = field(default=None, repr=False)
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    def __post_init__(self) -> None:
        if self.state is None:
            self.state = np.zeros(self.n, dtype=bool)

    def step(self) -> int:
        """One time step. Returns number of active neurons."""
        new = np.zeros(self.n, dtype=bool)
        # Branching: each active neuron causes sigma children.
        n_act = int(self.state.sum())
        if n_act > 0:
            n_children = self.rng.poisson(self.sigma * n_act)
            if n_children > 0:
                targets = self.rng.integers(0, self.n, size=n_children)
                new[targets] = True
        # External drive.
        new |= self.rng.uniform(size=self.n) < self.p_ext
        self.state = new
        return int(new.sum())


def measure_avalanches(net: BranchingNetwork, n_steps: int = 5000
                        ) -> tuple[list, list]:
    """Run network and detect avalanches (contiguous active periods).

    Returns (sizes, durations) lists.
    """
    sizes = []; durations = []
    cur_size = 0; cur_dur = 0
    for _ in range(n_steps):
        a = net.step()
        if a > 0:
            cur_size += a; cur_dur += 1
        else:
            if cur_dur > 0:
                sizes.append(cur_size); durations.append(cur_dur)
            cur_size = 0; cur_dur = 0
    if cur_dur > 0:
        sizes.append(cur_size); durations.append(cur_dur)
    return sizes, durations


def fit_power_law_exponent(values: list, x_min: int = 1) -> float:
    """ML estimator of power-law exponent for discrete data."""
    values = [v for v in values if v >= x_min]
    if len(values) < 5:
        return 0.0
    n = len(values)
    s = sum(np.log(v / (x_min - 0.5)) for v in values)
    return 1 + n / s
