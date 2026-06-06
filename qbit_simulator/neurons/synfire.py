"""Synfire chain (Abeles 1991).

Feed-forward chain of pools where each pool projects strongly to the
next. When the first pool spikes synchronously, a "packet" of spikes
propagates down the chain with high fidelity. A famous model for
sequential / motor-program activity.

Architecture:
  - L layers, each with M neurons.
  - Layer ℓ → ℓ+1 with all-to-all weights w.
  - First layer driven by an external pulse.

A synchronous pulse propagates if w·M / V_th is above a critical
threshold; otherwise it dies out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .lif import LIFPopulation


@dataclass
class SynfireChain:
    n_layers: int = 5
    layer_size: int = 20
    w: float = 0.2
    pop: LIFPopulation = field(default=None)

    def __post_init__(self) -> None:
        N = self.n_layers * self.layer_size
        if self.pop is None:
            self.pop = LIFPopulation(n=N, tau=20.0, t_refrac=2)

    def _connectivity(self) -> np.ndarray:
        N = self.n_layers * self.layer_size
        W = np.zeros((N, N))
        for ell in range(self.n_layers - 1):
            i0 = (ell + 1) * self.layer_size
            i1 = i0 + self.layer_size
            j0 = ell * self.layer_size
            j1 = j0 + self.layer_size
            W[i0:i1, j0:j1] = self.w
        return W

    def run(self, n_steps: int = 80, pulse_strength: float = 2.0,
            pulse_duration: int = 2) -> dict:
        W = self._connectivity()
        N = self.pop.n
        spike_history = np.zeros((n_steps, N), dtype=bool)
        last = np.zeros(N)
        for t in range(n_steps):
            ext = np.zeros(N)
            if t < pulse_duration:
                ext[:self.layer_size] = pulse_strength
            I = ext + W @ last
            spikes = self.pop.step(I, t=t)
            spike_history[t] = spikes
            last = spikes.astype(np.float64)
        # Per-layer spike count.
        layer_counts = np.zeros(self.n_layers, dtype=int)
        for ell in range(self.n_layers):
            j0 = ell * self.layer_size; j1 = j0 + self.layer_size
            layer_counts[ell] = int(spike_history[:, j0:j1].sum())
        return {"spikes": spike_history, "layer_counts": layer_counts}
