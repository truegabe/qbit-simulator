"""Spike-frequency adaptation (SFA).

Most cortical neurons fire at a rapidly decreasing rate to a sustained
stimulus — they "adapt". Mechanism: each spike opens slow potassium
channels (or a Ca²⁺-activated K current) that hyperpolarize the cell.

Simple LIF + adaptation:
  dV/dt  = (-V + V_rest + R·I - w) / tau
  dw/dt  = -w / tau_a
  on spike: w <- w + b
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AdaptingLIF:
    """LIF with spike-triggered adaptation current w."""
    n: int
    V_rest: float = 0.0
    V_th:   float = 1.0
    V_reset: float = 0.0
    tau:    float = 20.0
    tau_a:  float = 100.0
    R:      float = 1.0
    b:      float = 0.1
    t_refrac: int = 2
    V: np.ndarray = field(default=None, repr=False)
    w: np.ndarray = field(default=None, repr=False)
    refrac_until: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.V is None:
            self.V = np.full(self.n, self.V_rest)
        if self.w is None:
            self.w = np.zeros(self.n)
        if self.refrac_until is None:
            self.refrac_until = np.full(self.n, -1, dtype=np.int64)

    def step(self, I: np.ndarray, t: int, dt: float = 1.0) -> np.ndarray:
        active = t > self.refrac_until
        dV = dt / self.tau * (-(self.V - self.V_rest) + self.R * I - self.w)
        self.V = np.where(active, self.V + dV, self.V_reset)
        self.w += dt * (-self.w / self.tau_a)
        spikes = active & (self.V >= self.V_th)
        self.V = np.where(spikes, self.V_reset, self.V)
        self.w = np.where(spikes, self.w + self.b, self.w)
        self.refrac_until = np.where(spikes, t + self.t_refrac,
                                      self.refrac_until)
        return spikes
