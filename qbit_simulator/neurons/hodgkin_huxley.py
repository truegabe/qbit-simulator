"""Hodgkin-Huxley neuron model (1952 Nobel Prize).

The original biophysically-detailed spiking-neuron model with explicit
sodium and potassium ion channels.

    C_m dV/dt = -g_Na m^3 h (V - E_Na) - g_K n^4 (V - E_K)
                 - g_L (V - E_L) + I_ext

Gating variables m, h, n obey
    dx/dt = alpha_x(V) (1 - x) - beta_x(V) x

Default parameters use the classic squid-giant-axon values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---- rate functions (squid-axon parameters; mV, ms, μF/cm²) ----

def _alpha_m(V: np.ndarray) -> np.ndarray:
    # Avoid singularity at V = -40.
    d = V + 40.0
    return np.where(np.abs(d) < 1e-7, 1.0, 0.1 * d / (1 - np.exp(-d / 10)))


def _beta_m(V: np.ndarray) -> np.ndarray:
    return 4.0 * np.exp(-(V + 65.0) / 18.0)


def _alpha_h(V: np.ndarray) -> np.ndarray:
    return 0.07 * np.exp(-(V + 65.0) / 20.0)


def _beta_h(V: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))


def _alpha_n(V: np.ndarray) -> np.ndarray:
    d = V + 55.0
    return np.where(np.abs(d) < 1e-7, 0.1, 0.01 * d / (1 - np.exp(-d / 10)))


def _beta_n(V: np.ndarray) -> np.ndarray:
    return 0.125 * np.exp(-(V + 65.0) / 80.0)


@dataclass
class HHNeuron:
    """Single Hodgkin-Huxley neuron."""
    C_m:  float = 1.0     # μF/cm²
    g_Na: float = 120.0   # mS/cm²
    g_K:  float = 36.0
    g_L:  float = 0.3
    E_Na: float = 50.0    # mV
    E_K:  float = -77.0
    E_L:  float = -54.4

    V: float = -65.0
    m: float = 0.05
    h: float = 0.6
    n: float = 0.32
    _was_above: bool = False

    def currents(self) -> tuple[float, float, float]:
        I_Na = self.g_Na * self.m ** 3 * self.h * (self.V - self.E_Na)
        I_K  = self.g_K  * self.n ** 4 *           (self.V - self.E_K)
        I_L  = self.g_L *                           (self.V - self.E_L)
        return I_Na, I_K, I_L

    def step(self, I_ext: float, dt: float = 0.01) -> bool:
        """One Euler step. Returns True at the upstroke crossing 0 mV."""
        I_Na, I_K, I_L = self.currents()
        dV = (I_ext - I_Na - I_K - I_L) / self.C_m
        V_arr = np.array([self.V])
        am, bm = _alpha_m(V_arr)[0], _beta_m(V_arr)[0]
        ah, bh = _alpha_h(V_arr)[0], _beta_h(V_arr)[0]
        an, bn = _alpha_n(V_arr)[0], _beta_n(V_arr)[0]
        self.m += dt * (am * (1 - self.m) - bm * self.m)
        self.h += dt * (ah * (1 - self.h) - bh * self.h)
        self.n += dt * (an * (1 - self.n) - bn * self.n)
        self.V += dt * dV
        spike = (not self._was_above) and self.V >= 0.0
        self._was_above = self.V >= 0.0
        return spike


def run_hh(neuron: HHNeuron, I_func, n_steps: int,
           dt: float = 0.01) -> dict:
    V = np.zeros(n_steps)
    spikes = np.zeros(n_steps, dtype=bool)
    for t in range(n_steps):
        I = I_func(t) if callable(I_func) else I_func
        spikes[t] = neuron.step(I, dt=dt)
        V[t] = neuron.V
    return {"V": V, "spikes": spikes, "n_spikes": int(spikes.sum())}
