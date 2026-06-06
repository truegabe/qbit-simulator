"""Kuramoto coupled oscillators.

    dθ_i/dt = ω_i + (K/N) sum_j sin(θ_j - θ_i)

A canonical model of phase synchronization in networks of oscillators.
At critical coupling K_c, a phase transition to synchrony occurs.

Order parameter:
    r e^{iψ} = (1/N) sum_j e^{iθ_j}
r = 0: incoherent. r = 1: full sync.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Kuramoto:
    n: int = 100
    K: float = 1.0
    sigma_omega: float = 0.5
    theta: np.ndarray = field(default=None, repr=False)
    omega: np.ndarray = field(default=None, repr=False)
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    def __post_init__(self) -> None:
        if self.theta is None:
            self.theta = self.rng.uniform(-np.pi, np.pi, self.n)
        if self.omega is None:
            self.omega = self.rng.normal(0, self.sigma_omega, self.n)

    def order_parameter(self) -> tuple[float, float]:
        z = np.exp(1j * self.theta).mean()
        return float(np.abs(z)), float(np.angle(z))

    def step(self, dt: float = 0.05) -> tuple[float, float]:
        # Vectorized coupling sum.
        # sin(θ_j - θ_i) = sin θ_j cos θ_i - cos θ_j sin θ_i
        s = np.sin(self.theta); c = np.cos(self.theta)
        sum_s = s.sum(); sum_c = c.sum()
        coupling = (sum_s * c - sum_c * s) / self.n
        self.theta += dt * (self.omega + self.K * coupling)
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi
        return self.order_parameter()

    def run(self, n_steps: int = 500, dt: float = 0.05) -> np.ndarray:
        out = np.zeros(n_steps)
        for t in range(n_steps):
            r, _ = self.step(dt=dt)
            out[t] = r
        return out
