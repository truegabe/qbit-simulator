"""Astrocyte / glial modulation.

Astrocytes (a major type of glial cell) wrap synapses and dynamically
modulate their efficacy via calcium signaling and gliotransmitter
release. A simple model:

  - Tripartite synapse: pre + post + astrocyte.
  - Pre-synaptic activity raises astrocyte intracellular Ca²⁺ via IP3.
  - When Ca²⁺ crosses a threshold, the astrocyte releases gliotransmitter
    (e.g. glutamate or ATP) that modulates the synaptic weight.

We implement a slow Ca²⁺ trace per synapse, threshold-triggered weight
boost, and slow decay back to baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AstrocyteModulator:
    """A bank of astrocyte processes wrapping `n` synapses."""
    n: int
    tau_Ca: float = 200.0
    Ca_threshold: float = 1.0
    boost: float = 0.2
    tau_boost: float = 500.0
    Ca: np.ndarray = field(default=None, repr=False)
    gain: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.Ca is None:
            self.Ca = np.zeros(self.n)
        if self.gain is None:
            self.gain = np.ones(self.n)

    def step(self, pre_activity: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """Receive pre-synaptic activity, update Ca²⁺ and synaptic gain.

        Returns the current per-synapse gain (multiplier on weights).
        """
        # IP3 → Ca²⁺ leak/influx.
        self.Ca += dt * (-self.Ca / self.tau_Ca + pre_activity)
        # On Ca²⁺ threshold crossing, boost gain.
        spike_Ca = self.Ca >= self.Ca_threshold
        self.gain = np.where(spike_Ca, self.gain + self.boost, self.gain)
        self.Ca = np.where(spike_Ca, 0.0, self.Ca)
        # Gain decays back to 1 (baseline).
        self.gain += dt * (1.0 - self.gain) / self.tau_boost
        return self.gain


@dataclass
class TripartiteSynapse:
    """A synapse with pre, post, and astrocyte modulation."""
    n: int
    base_weight: float = 1.0
    W: np.ndarray = field(default=None, repr=False)
    astro: AstrocyteModulator = field(default=None)

    def __post_init__(self) -> None:
        if self.W is None:
            self.W = np.full(self.n, self.base_weight)
        if self.astro is None:
            self.astro = AstrocyteModulator(n=self.n)

    def transmit(self, pre_spikes: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """Pre-spikes go in, modulated post-synaptic currents come out."""
        gain = self.astro.step(pre_spikes.astype(np.float64), dt=dt)
        return self.W * gain * pre_spikes.astype(np.float64)
