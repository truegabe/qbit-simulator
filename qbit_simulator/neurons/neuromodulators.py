"""Neuromodulator system: dopamine, serotonin, noradrenaline, ACh.

Different neuromodulators play different roles:
  - DA  (dopamine):       reward prediction error, motivation
  - 5HT (serotonin):      patience, mood, aversive value
  - NE  (noradrenaline):  arousal, gain modulation, novelty
  - ACh (acetylcholine):  attention, signal-to-noise, plasticity

Each is modelled as a scalar level with its own:
  - Slow tonic baseline.
  - Phasic events on a triggering signal (reward, punishment, surprise,
    attention).

The combined neuromodulator state affects downstream learning and
gain. Typical use: pass these levels into plasticity rules
(see three_factor.py) or as gain multipliers on attention.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class NeuromodulatorSystem:
    """Four neuromodulators with simple leaky-integrator dynamics."""
    tau_DA:  float = 200.0
    tau_5HT: float = 1000.0
    tau_NE:  float = 100.0
    tau_ACh: float = 150.0

    tonic_DA:  float = 0.2
    tonic_5HT: float = 0.3
    tonic_NE:  float = 0.2
    tonic_ACh: float = 0.2

    DA:  float = 0.2
    HT:  float = 0.3
    NE:  float = 0.2
    ACh: float = 0.2

    def step(self, dt: float = 1.0,
              reward: float = 0.0, aversion: float = 0.0,
              novelty: float = 0.0, attention: float = 0.0) -> dict:
        """Tonic baseline + phasic responses."""
        self.DA  += dt * (self.tonic_DA  - self.DA)  / self.tau_DA  + reward
        self.HT  += dt * (self.tonic_5HT - self.HT)  / self.tau_5HT + aversion
        self.NE  += dt * (self.tonic_NE  - self.NE)  / self.tau_NE  + novelty
        self.ACh += dt * (self.tonic_ACh - self.ACh) / self.tau_ACh + attention
        return self.levels()

    def levels(self) -> dict:
        return {"DA": self.DA, "5HT": self.HT, "NE": self.NE, "ACh": self.ACh}

    def learning_gain(self) -> float:
        """How much plasticity should the system permit right now?

        High DA, ACh and NE all boost plasticity; high 5HT suppresses
        impulsive learning (encourages waiting).
        """
        return float(self.DA * (1 + self.ACh) * (1 + self.NE) / (1 + self.HT))

    def signal_to_noise_gain(self) -> float:
        """NE / ACh-driven gain (Aston-Jones & Cohen). Returns multiplier."""
        return float(1.0 + 0.5 * self.NE + 0.3 * self.ACh)
