"""Thalamocortical relay loop with attention gating.

Sensory information traverses thalamus before reaching cortex. The
thalamic-reticular nucleus (TRN) provides inhibitory gating that
attention can modulate ("searchlight" hypothesis, Crick 1984).

Model:
  - Sensory input → thalamus → cortex.
  - Cortex has reciprocal connections back to thalamus AND to TRN.
  - TRN inhibits thalamus (reduces relay gain).
  - Attention signal (top-down) suppresses TRN for attended channels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Thalamocortical:
    n_channels: int = 8
    tau: float = 10.0
    relay_gain: float = 1.0
    trn_strength: float = 0.5
    cortex_feedback: float = 0.2
    thal: np.ndarray = field(default=None, repr=False)
    cortex: np.ndarray = field(default=None, repr=False)
    trn: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.thal is None:
            self.thal = np.zeros(self.n_channels)
        if self.cortex is None:
            self.cortex = np.zeros(self.n_channels)
        if self.trn is None:
            self.trn = np.zeros(self.n_channels)

    def step(self, sensory: np.ndarray, attention: np.ndarray | None = None,
              dt: float = 1.0) -> dict:
        """One step. attention: per-channel ∈ [0, 1], 1=attended (TRN suppressed)."""
        if attention is None:
            attention = np.ones(self.n_channels) * 0.5
        # TRN integrates cortex output minus attentional suppression.
        dtrn = (-self.trn + self.cortex - 2.0 * attention) / self.tau
        # Thalamus integrates sensory minus TRN inhibition.
        dthal = (-self.thal + self.relay_gain * sensory
                  - self.trn_strength * np.maximum(self.trn, 0)) / self.tau
        # Cortex integrates thalamic relay plus modest self-recurrence.
        dctx = (-self.cortex + np.maximum(self.thal, 0)
                 + self.cortex_feedback * np.maximum(self.cortex, 0)) / self.tau
        self.trn += dt * dtrn
        self.thal += dt * dthal
        self.cortex += dt * dctx
        return {"thal": self.thal.copy(), "cortex": self.cortex.copy(),
                "trn": self.trn.copy()}

    def run(self, sensory: np.ndarray, n_steps: int = 200,
             attention: np.ndarray | None = None,
             dt: float = 1.0) -> dict:
        T = np.zeros((n_steps, self.n_channels))
        C = np.zeros((n_steps, self.n_channels))
        for t in range(n_steps):
            out = self.step(sensory, attention=attention, dt=dt)
            T[t] = out["thal"]; C[t] = out["cortex"]
        return {"thalamus": T, "cortex": C}
