"""Tempotron (Gütig & Sompolinsky, 2006).

A temporal-pattern-classifier neuron that learns to spike on "positive"
spatiotemporal spike patterns and stay silent on "negative" ones.

Voltage:
    V(t) = sum_i w_i sum_{t_i^k < t} K(t - t_i^k)

with kernel K(s) = V_norm · (exp(-s/tau_m) - exp(-s/tau_s)),
where tau_m > tau_s. The neuron fires if V crosses V_th anywhere.

Learning rule (per pattern):
  - If target=1 and didn't spike: Δw_i = +λ · sum_{t_i^k ≤ t_max} K(t_max - t_i^k)
    (potentiate at t_max = argmax V).
  - If target=0 and did spike: Δw_i = −λ · sum_{t_i^k ≤ t*} K(t* - t_i^k)
    where t* = first threshold crossing.
  - Otherwise: no update.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Tempotron:
    n_inputs: int
    T: int = 100          # presentation duration (steps)
    tau_m: float = 15.0
    tau_s: float = 3.75
    V_th: float = 1.0
    eta: float = 0.001
    w: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.w is None:
            self.w = np.random.default_rng(0).normal(
                0.0, 0.01, size=self.n_inputs)
        # Normalization constant for the kernel.
        self.V_norm = self._V_norm()

    def _V_norm(self) -> float:
        # Peak of (exp(-t/tau_m) - exp(-t/tau_s)) is at
        # t* = tau_m tau_s / (tau_m - tau_s) · ln(tau_m/tau_s).
        ratio = self.tau_m / self.tau_s
        t_star = (self.tau_m * self.tau_s / (self.tau_m - self.tau_s)) * np.log(ratio)
        peak = np.exp(-t_star / self.tau_m) - np.exp(-t_star / self.tau_s)
        return 1.0 / max(peak, 1e-9)

    def _kernel(self, s: np.ndarray) -> np.ndarray:
        out = np.zeros_like(s, dtype=np.float64)
        mask = s >= 0
        out[mask] = self.V_norm * (np.exp(-s[mask] / self.tau_m)
                                    - np.exp(-s[mask] / self.tau_s))
        return out

    def voltage_trace(self, spike_times: list) -> np.ndarray:
        """spike_times: list of arrays of input spike times (one array per
        input). Returns V(t) over [0, T)."""
        t_arr = np.arange(self.T, dtype=np.float64)
        V = np.zeros(self.T)
        for i, ts in enumerate(spike_times):
            for tk in ts:
                V += self.w[i] * self._kernel(t_arr - tk)
        return V

    def classify(self, spike_times: list) -> tuple[bool, np.ndarray]:
        V = self.voltage_trace(spike_times)
        return bool(np.any(V >= self.V_th)), V

    def train_one(self, spike_times: list, target: int) -> bool:
        """One online update. Returns True if classification was correct."""
        fired, V = self.classify(spike_times)
        if target == 1 and not fired:
            t_star = int(np.argmax(V))
            for i, ts in enumerate(spike_times):
                self.w[i] += self.eta * sum(
                    self._kernel(np.array([t_star - tk]))[0]
                    for tk in ts if tk <= t_star)
            return False
        if target == 0 and fired:
            crossings = np.where(V >= self.V_th)[0]
            t_star = int(crossings[0])
            for i, ts in enumerate(spike_times):
                self.w[i] -= self.eta * sum(
                    self._kernel(np.array([t_star - tk]))[0]
                    for tk in ts if tk <= t_star)
            return False
        return True
