"""Three-factor learning rules.

Classical Hebbian learning uses two factors: pre-synaptic activity and
post-synaptic activity. Three-factor rules add a third "gating" signal
M(t) — typically a neuromodulator (dopamine, ACh, noradrenaline) that
permits or vetoes plasticity:

    Δw_ij(t) = M(t) · F(pre_j, post_i)

This is the standard way to combine local Hebbian eligibility traces
with global reward/error signals — the substrate for reinforcement
learning in the brain.

Eligibility trace:
    de_ij/dt = -e_ij/tau_e + F(pre_j, post_i)
Weight update:
    dw_ij/dt = eta · M(t) · e_ij(t)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ThreeFactorLearner:
    """Three-factor learning with Hebbian eligibility traces."""
    n_pre: int
    n_post: int
    tau_e: float = 100.0
    eta: float = 0.001
    e: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.e is None:
            self.e = np.zeros((self.n_post, self.n_pre))

    def step(self, W: np.ndarray, pre: np.ndarray, post: np.ndarray,
              modulator: float, dt: float = 1.0) -> np.ndarray:
        """Update eligibility, then apply gated weight change."""
        # Decay eligibility.
        self.e *= np.exp(-dt / self.tau_e)
        # Hebbian eligibility increment.
        self.e += np.outer(post, pre)
        # Modulated weight update.
        W = W + dt * self.eta * modulator * self.e
        return W

    def reset(self) -> None:
        self.e[:] = 0.0


@dataclass
class DopamineModulator:
    """A dopamine signal generator: tonic level + phasic bursts on reward.

    Tonic baseline tau_DA → 0 (slow decay). On reward delivery, a
    short positive transient is added.
    """
    tonic: float = 0.0
    tau_DA: float = 200.0
    phasic_amp: float = 1.0
    level: float = field(default=None)

    def __post_init__(self) -> None:
        if self.level is None:
            self.level = self.tonic

    def step(self, dt: float, reward: float = 0.0) -> float:
        self.level += dt * (self.tonic - self.level) / self.tau_DA
        self.level += reward * self.phasic_amp
        return self.level
