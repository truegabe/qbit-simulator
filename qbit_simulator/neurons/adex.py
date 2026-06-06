"""Adaptive Exponential Integrate-and-Fire (AdEx) neuron.

Brette & Gerstner (2005). Captures spike initiation via exponential
upswing plus an adaptation variable w that decays slowly. The model
can reproduce most cortical firing patterns with just 5 parameters.

    C dV/dt = -g_L (V - E_L) + g_L Δ_T exp((V - V_T)/Δ_T) - w + I
    τ_w dw/dt = a (V - E_L) - w

Spike when V crosses V_peak: V <- V_reset, w <- w + b.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AdExNeuron:
    """Single AdEx neuron with default cortical-RS parameters."""
    C:       float = 281.0    # pF
    g_L:     float = 30.0     # nS
    E_L:     float = -70.6    # mV
    V_T:     float = -50.4
    Delta_T: float = 2.0
    tau_w:   float = 144.0    # ms
    a:       float = 4.0      # nS
    b:       float = 0.0805   # nA
    V_reset: float = -70.6
    V_peak:  float = 0.0

    V: float = field(default=None)
    w: float = 0.0

    def __post_init__(self) -> None:
        if self.V is None:
            self.V = self.E_L

    def step(self, I: float, dt: float = 0.1) -> bool:
        """One Euler step, dt in ms. Returns True on spike."""
        # Clip exponential to avoid overflow.
        arg = (self.V - self.V_T) / self.Delta_T
        exp_term = self.g_L * self.Delta_T * np.exp(min(arg, 50.0))
        dV = (-self.g_L * (self.V - self.E_L) + exp_term - self.w + I) / self.C
        dw = (self.a * (self.V - self.E_L) - self.w) / self.tau_w
        self.V += dt * dV
        self.w += dt * dw
        if self.V >= self.V_peak:
            self.V = self.V_reset
            self.w += self.b
            return True
        return False


@dataclass
class AdExPopulation:
    """Vectorized AdEx population."""
    n: int
    C:       float = 281.0
    g_L:     float = 30.0
    E_L:     float = -70.6
    V_T:     float = -50.4
    Delta_T: float = 2.0
    tau_w:   float = 144.0
    a:       float = 4.0
    b:       float = 0.0805
    V_reset: float = -70.6
    V_peak:  float = 0.0

    V: np.ndarray = field(default=None, repr=False)
    w: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.V is None:
            self.V = np.full(self.n, self.E_L)
        if self.w is None:
            self.w = np.zeros(self.n)

    def step(self, I: np.ndarray, dt: float = 0.1) -> np.ndarray:
        arg = np.clip((self.V - self.V_T) / self.Delta_T, None, 50.0)
        exp_term = self.g_L * self.Delta_T * np.exp(arg)
        dV = (-self.g_L * (self.V - self.E_L) + exp_term - self.w + I) / self.C
        dw = (self.a * (self.V - self.E_L) - self.w) / self.tau_w
        self.V += dt * dV
        self.w += dt * dw
        spikes = self.V >= self.V_peak
        self.V = np.where(spikes, self.V_reset, self.V)
        self.w = np.where(spikes, self.w + self.b, self.w)
        return spikes


def run_adex(neuron: AdExNeuron, I_func, n_steps: int,
             dt: float = 0.1) -> dict:
    V = np.zeros(n_steps); w = np.zeros(n_steps)
    spikes = np.zeros(n_steps, dtype=bool)
    for t in range(n_steps):
        I = I_func(t) if callable(I_func) else I_func
        spikes[t] = neuron.step(I, dt=dt)
        V[t] = neuron.V; w[t] = neuron.w
    return {"V": V, "w": w, "spikes": spikes, "n_spikes": int(spikes.sum())}
