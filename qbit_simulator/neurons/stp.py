"""Short-term plasticity (Tsodyks-Markram model).

Synapses don't just have a static weight — their efficacy changes on
the time scale of hundreds of ms in response to recent activity.

Two phenomenological state variables per synapse:
  u : "use" / release probability (facilitation)
  x : fraction of resources available (depression)

On a pre-synaptic spike at time t:
  u <- u + U (1 - u)            # facilitation jump
  released = u · x               # post-synaptic effect
  x <- x - released              # vesicle depletion

Between spikes:
  du/dt = -(u - U)/tau_f
  dx/dt = (1 - x)/tau_d

(U, tau_d, tau_f) determine the synapse type:
  Depressing  : (0.5, 800, 0)
  Facilitating: (0.15, 100, 1500)
  Pseudo-static: (0.5, 100, 100)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TsodyksMarkramSynapse:
    """Single Tsodyks-Markram synapse."""
    U: float = 0.5
    tau_d: float = 800.0
    tau_f: float = 0.0   # 0 = no facilitation
    A: float = 1.0       # baseline post-synaptic amplitude

    u: float = field(default=None)
    x: float = 1.0

    def __post_init__(self) -> None:
        if self.u is None:
            self.u = self.U

    def step(self, dt: float, pre_spike: bool) -> float:
        """Advance dt; if pre_spike, emit released = A * u * x and update.

        Returns the released amplitude this step (0 if no spike).
        """
        # Decay between spikes.
        if self.tau_f > 0:
            self.u += dt * (self.U - self.u) / self.tau_f
        self.x += dt * (1.0 - self.x) / self.tau_d
        if not pre_spike:
            return 0.0
        # Facilitation: bump u toward 1.
        if self.tau_f > 0:
            self.u = self.u + self.U * (1 - self.u)
        else:
            self.u = self.U
        released = self.A * self.u * self.x
        self.x = self.x - self.u * self.x
        return released


@dataclass
class STPPopulation:
    """Vectorized STP on a population of N pre-synaptic neurons."""
    n: int
    U: float = 0.5
    tau_d: float = 800.0
    tau_f: float = 0.0
    A: float = 1.0
    u: np.ndarray = field(default=None, repr=False)
    x: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.u is None:
            self.u = np.full(self.n, self.U)
        if self.x is None:
            self.x = np.ones(self.n)

    def step(self, dt: float, pre_spikes: np.ndarray) -> np.ndarray:
        """One step. Returns released amplitudes per pre-neuron."""
        if self.tau_f > 0:
            self.u += dt * (self.U - self.u) / self.tau_f
        self.x += dt * (1.0 - self.x) / self.tau_d
        # On pre-spike, do the discrete update.
        spike_mask = pre_spikes.astype(bool)
        if self.tau_f > 0:
            self.u = np.where(
                spike_mask, self.u + self.U * (1 - self.u), self.u
            )
        else:
            self.u = np.where(spike_mask, self.U, self.u)
        released = np.where(spike_mask, self.A * self.u * self.x, 0.0)
        self.x = np.where(spike_mask, self.x - self.u * self.x, self.x)
        return released


def depressing_synapse(**kw) -> TsodyksMarkramSynapse:
    return TsodyksMarkramSynapse(U=0.5, tau_d=800.0, tau_f=0.0, **kw)


def facilitating_synapse(**kw) -> TsodyksMarkramSynapse:
    return TsodyksMarkramSynapse(U=0.15, tau_d=100.0, tau_f=1500.0, **kw)


def pseudo_static_synapse(**kw) -> TsodyksMarkramSynapse:
    return TsodyksMarkramSynapse(U=0.5, tau_d=100.0, tau_f=100.0, **kw)
