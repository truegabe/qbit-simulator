"""Synaptic Homeostasis Hypothesis (Tononi & Cirelli 2014).

During wakefulness, synapses potentiate (net LTP) from new learning.
During sleep, ALL synapses are downscaled proportionally — preserving
RELATIVE strengths while bringing total synaptic load back down.

Mechanism: w <- max(0, w - σ · w · sleep_pressure).
This maintains signal-to-noise (strong synapses stay strong relative
to weak ones), saves energy, and prevents runaway potentiation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SHYNetwork:
    n: int
    W: np.ndarray = field(default=None, repr=False)
    sleep_pressure: float = 0.0   # accumulates during wake
    downscale_rate: float = 0.001

    def __post_init__(self) -> None:
        if self.W is None:
            self.W = np.zeros((self.n, self.n))

    def wake_step(self, hebbian_input: np.ndarray) -> None:
        """LTP from learning + sleep pressure accumulates."""
        self.W += hebbian_input
        self.sleep_pressure += float(np.abs(hebbian_input).sum())

    def sleep_step(self, dt: float = 1.0) -> None:
        """Multiplicative downscaling of all synapses."""
        scale = max(1 - self.downscale_rate * dt * self.sleep_pressure, 0.5)
        self.W *= scale
        self.sleep_pressure *= 0.99   # pressure dissipates

    def total_weight(self) -> float:
        return float(np.abs(self.W).sum())


def sleep_cycle(net: SHYNetwork, n_steps: int = 100, dt: float = 1.0
                 ) -> np.ndarray:
    """Run pure-sleep cycle, return total weight trajectory."""
    out = np.zeros(n_steps)
    for t in range(n_steps):
        net.sleep_step(dt=dt)
        out[t] = net.total_weight()
    return out
