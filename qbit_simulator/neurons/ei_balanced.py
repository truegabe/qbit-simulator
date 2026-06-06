"""Excitatory-Inhibitory balanced network (Brunel 2000).

A canonical model of asynchronous cortical activity. Two populations:
  - N_E excitatory neurons
  - N_I inhibitory neurons (typically N_E = 4 N_I)

Connectivity is sparse random with probability p. Excitatory weight is
J; inhibitory weight is -g*J (with g > 1 so inhibition dominates per
synapse, balancing the larger E population).

When g and external drive ν_ext are chosen well, the network settles
into an AI (asynchronous irregular) state: low firing rates, CV ≈ 1,
weak cross-correlations — the hallmark of cortical activity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .lif import LIFPopulation


@dataclass
class EIBalancedNetwork:
    """Brunel-style sparse random E-I network."""
    N_E: int = 800
    N_I: int = 200
    p:   float = 0.1
    J:   float = 0.1
    g:   float = 5.0
    nu_ext: float = 0.5         # external Poisson rate per neuron per step
    pop: LIFPopulation = field(default=None)
    W:   np.ndarray = field(default=None, repr=False)
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    def __post_init__(self) -> None:
        N = self.N_E + self.N_I
        if self.pop is None:
            self.pop = LIFPopulation(n=N, tau=20.0, t_refrac=2)
        if self.W is None:
            W = np.zeros((N, N))
            # Excitatory presynaptic columns: cols [0, N_E)
            mask_E = self.rng.uniform(size=(N, self.N_E)) < self.p
            W[:, :self.N_E] = mask_E * self.J
            # Inhibitory presynaptic columns: cols [N_E, N)
            mask_I = self.rng.uniform(size=(N, self.N_I)) < self.p
            W[:, self.N_E:] = mask_I * (-self.g * self.J)
            # No self-connections.
            np.fill_diagonal(W, 0.0)
            self.W = W

    def run(self, n_steps: int = 500) -> dict:
        N = self.pop.n
        spike_history = np.zeros((n_steps, N), dtype=bool)
        last_spikes = np.zeros(N)
        for t in range(n_steps):
            recurrent = self.W @ last_spikes
            ext = (self.rng.uniform(size=N) < self.nu_ext).astype(np.float64) * self.J * 20
            I_tot = recurrent + ext
            spikes = self.pop.step(I_tot, t=t)
            spike_history[t] = spikes
            last_spikes = spikes.astype(np.float64)
        rates_E = spike_history[:, :self.N_E].mean(axis=0)
        rates_I = spike_history[:, self.N_E:].mean(axis=0)
        return {
            "spikes": spike_history,
            "rates_E": rates_E, "rates_I": rates_I,
            "mean_rate_E": float(rates_E.mean()),
            "mean_rate_I": float(rates_I.mean()),
        }
