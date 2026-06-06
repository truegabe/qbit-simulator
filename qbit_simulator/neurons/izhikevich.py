"""Izhikevich neuron model.

Eugene Izhikevich's 2003 model: as biologically rich as Hodgkin-Huxley
but only 2 ODEs, so cheap enough for million-neuron sims.

    dv/dt = 0.04 v^2 + 5 v + 140 - u + I
    du/dt = a (b v - u)

Spike condition: v >= 30 mV. On spike: v <- c, u <- u + d.

The (a, b, c, d) tuple selects a firing regime:
  - Regular spiking   : (0.02, 0.2, -65, 8)
  - Intrinsically bursting: (0.02, 0.2, -55, 4)
  - Chattering        : (0.02, 0.2, -50, 2)
  - Fast spiking      : (0.1,  0.2, -65, 2)
  - Low-threshold spiking: (0.02, 0.25, -65, 2)
  - Resonator         : (0.1,  0.26, -65, 2)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


REGIMES = {
    "RS":  (0.02, 0.20, -65.0, 8.0),
    "IB":  (0.02, 0.20, -55.0, 4.0),
    "CH":  (0.02, 0.20, -50.0, 2.0),
    "FS":  (0.10, 0.20, -65.0, 2.0),
    "LTS": (0.02, 0.25, -65.0, 2.0),
    "RZ":  (0.10, 0.26, -65.0, 2.0),
}


@dataclass
class IzhikevichNeuron:
    """Single Izhikevich neuron."""
    a: float = 0.02
    b: float = 0.20
    c: float = -65.0
    d: float = 8.0
    v: float = -65.0
    u: float = field(default=None)

    def __post_init__(self) -> None:
        if self.u is None:
            self.u = self.b * self.v

    @classmethod
    def from_regime(cls, name: str) -> "IzhikevichNeuron":
        a, b, c, d = REGIMES[name]
        return cls(a=a, b=b, c=c, d=d, v=c)

    def step(self, I: float, dt: float = 0.5) -> bool:
        """Half-step Euler, then check spike. Returns True if spiked."""
        # Two half-steps for v (Izhikevich's recommendation for stability).
        self.v += 0.5 * dt * (0.04 * self.v * self.v + 5 * self.v + 140 - self.u + I)
        self.v += 0.5 * dt * (0.04 * self.v * self.v + 5 * self.v + 140 - self.u + I)
        self.u += dt * self.a * (self.b * self.v - self.u)
        if self.v >= 30.0:
            self.v = self.c
            self.u += self.d
            return True
        return False


@dataclass
class IzhikevichPopulation:
    """Vectorized population."""
    n: int
    a: np.ndarray = field(default=None, repr=False)
    b: np.ndarray = field(default=None, repr=False)
    c: np.ndarray = field(default=None, repr=False)
    d: np.ndarray = field(default=None, repr=False)
    v: np.ndarray = field(default=None, repr=False)
    u: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.a is None:
            self.a = np.full(self.n, 0.02)
        if self.b is None:
            self.b = np.full(self.n, 0.20)
        if self.c is None:
            self.c = np.full(self.n, -65.0)
        if self.d is None:
            self.d = np.full(self.n, 8.0)
        if self.v is None:
            self.v = self.c.copy()
        if self.u is None:
            self.u = self.b * self.v

    @classmethod
    def from_regime(cls, n: int, name: str) -> "IzhikevichPopulation":
        a, b, c, d = REGIMES[name]
        return cls(
            n=n,
            a=np.full(n, a), b=np.full(n, b),
            c=np.full(n, c), d=np.full(n, d),
            v=np.full(n, c), u=np.full(n, b * c),
        )

    def step(self, I: np.ndarray, dt: float = 0.5) -> np.ndarray:
        """One step. Returns boolean spike array."""
        for _ in range(2):
            self.v += 0.5 * dt * (0.04 * self.v ** 2 + 5 * self.v + 140 - self.u + I)
        self.u += dt * self.a * (self.b * self.v - self.u)
        spikes = self.v >= 30.0
        self.v = np.where(spikes, self.c, self.v)
        self.u = np.where(spikes, self.u + self.d, self.u)
        return spikes


def run_izhikevich(neuron: IzhikevichNeuron, I_func, n_steps: int,
                   dt: float = 0.5) -> dict:
    """Run simulation. I_func is callable(t) or scalar."""
    vs = np.zeros(n_steps)
    spikes = np.zeros(n_steps, dtype=bool)
    for t in range(n_steps):
        I = I_func(t) if callable(I_func) else I_func
        spikes[t] = neuron.step(I, dt=dt)
        vs[t] = neuron.v
    return {"v": vs, "spikes": spikes, "n_spikes": int(spikes.sum())}
