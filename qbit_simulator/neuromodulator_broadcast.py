"""Neuromodulator broadcast -- global chemical signalling system.

Four neuromodulatory systems are modelled:

  Dopamine (DA)      -- reward prediction error, gain modulation, learning rate
  Serotonin (5HT)    -- patience, risk aversion, mood (exploration vs exploit)
  Acetylcholine (ACh)-- attention, signal-to-noise ratio, memory encoding gate
  Norepinephrine (NE)-- arousal, alertness, global gain, noise threshold

These are NOT point-to-point signals.  Each is a broadcast:
one source nucleus (VTA, raphe, basal forebrain, locus coeruleus)
projects diffusely to nearly ALL brain regions simultaneously.

Effect on a receiving brain module:
  DA  high  -> higher learning rate, stronger reward-driven updates
  DA  low   -> reduced plasticity (stable, frozen weights)
  5HT high  -> more patient (lower discount rate gamma), less exploration
  5HT low   -> impulsive, higher exploration
  ACh high  -> sharper signal-to-noise (suppress background noise)
  ACh low   -> diffuse, noisy representations
  NE  high  -> alert, sensitive to weak signals, global gain up
  NE  low   -> sluggish, high threshold, ignores weak inputs

Classes
-------
  NeuromodulatorState   -- current levels of all four modulators
  NeuromodulatorSystem  -- source: generates levels from events
  ModulatableParams     -- effect parameters a brain module exposes
  NeuromodulatorBroadcast -- sends state to all registered modules
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# NeuromodulatorState
# ---------------------------------------------------------------------------

@dataclass
class NeuromodulatorState:
    """Current tonic levels of all four modulators.  All in [0, 1]."""
    DA:  float = 0.5   # dopamine
    ACh: float = 0.5   # acetylcholine
    NE:  float = 0.5   # norepinephrine
    HT5: float = 0.5   # serotonin (5-HT)

    def as_array(self) -> np.ndarray:
        return np.array([self.DA, self.ACh, self.NE, self.HT5])

    def from_array(self, arr: np.ndarray) -> None:
        self.DA, self.ACh, self.NE, self.HT5 = np.clip(arr, 0, 1)

    def __repr__(self) -> str:
        return (f"NeuromodulatorState("
                f"DA={self.DA:.3f}, ACh={self.ACh:.3f}, "
                f"NE={self.NE:.3f}, 5HT={self.HT5:.3f})")


# ---------------------------------------------------------------------------
# NeuromodulatorSystem  (source nucleus)
# ---------------------------------------------------------------------------

@dataclass
class NeuromodulatorSystem:
    """Models the source nuclei that generate neuromodulator levels.

    Driven by:
      reward_signal   -> raises DA (positive RPE) or lowers DA (negative RPE)
      novelty_signal  -> raises NE and ACh
      threat_signal   -> raises NE, lowers 5HT
      satiety_signal  -> raises 5HT, lowers DA slightly

    Each level decays toward its baseline with time constant tau.
    """
    baseline: NeuromodulatorState = field(
        default_factory=lambda: NeuromodulatorState(
            DA=0.5, ACh=0.5, NE=0.3, HT5=0.5))
    tau:  float = 10.0   # time-steps to return to baseline
    state: NeuromodulatorState = field(
        default_factory=lambda: NeuromodulatorState())

    def __post_init__(self) -> None:
        self.state = NeuromodulatorState(
            DA=self.baseline.DA, ACh=self.baseline.ACh,
            NE=self.baseline.NE, HT5=self.baseline.HT5)

    def step(self, dt: float = 1.0) -> None:
        """Decay all levels back toward baseline."""
        alpha = dt / self.tau
        self.state.DA  += alpha * (self.baseline.DA  - self.state.DA)
        self.state.ACh += alpha * (self.baseline.ACh - self.state.ACh)
        self.state.NE  += alpha * (self.baseline.NE  - self.state.NE)
        self.state.HT5 += alpha * (self.baseline.HT5 - self.state.HT5)

    def reward_signal(self, rpe: float) -> None:
        """Reward prediction error -> DA transient."""
        self.state.DA = float(np.clip(self.state.DA + 0.3 * rpe, 0, 1))

    def novelty_signal(self, novelty: float) -> None:
        """Novel stimulus -> NE + ACh transient."""
        self.state.NE  = float(np.clip(self.state.NE  + 0.2 * novelty, 0, 1))
        self.state.ACh = float(np.clip(self.state.ACh + 0.15 * novelty, 0, 1))

    def threat_signal(self, threat: float) -> None:
        """Threat -> NE up, 5HT down."""
        self.state.NE  = float(np.clip(self.state.NE  + 0.3 * threat, 0, 1))
        self.state.HT5 = float(np.clip(self.state.HT5 - 0.2 * threat, 0, 1))

    def satiety_signal(self, satiety: float) -> None:
        """Satiety -> 5HT up, DA slightly down."""
        self.state.HT5 = float(np.clip(self.state.HT5 + 0.2 * satiety, 0, 1))
        self.state.DA  = float(np.clip(self.state.DA  - 0.05 * satiety, 0, 1))

    def set(self, **kwargs) -> None:
        """Directly set any level: set(DA=0.8, NE=0.6)."""
        for k, v in kwargs.items():
            setattr(self.state, k, float(np.clip(v, 0, 1)))


# ---------------------------------------------------------------------------
# ModulatableParams  -- effect parameters a brain module exposes
# ---------------------------------------------------------------------------

@dataclass
class ModulatableParams:
    """Translated effect of neuromodulator levels on a brain module.

    These are the knobs that a receiving region adjusts in response
    to the broadcast.  Brain modules can read these directly.

    All values are dimensionless multipliers around 1.0 (neutral).
    """
    learning_rate_scale: float = 1.0   # DA:  high DA -> higher LR
    gain:                float = 1.0   # NE:  high NE -> higher gain
    noise_threshold:     float = 0.0   # ACh: high ACh -> suppress noise
    discount_rate:       float = 0.95  # 5HT: high 5HT -> more patient (higher gamma)
    exploration:         float = 0.1   # 5HT: low 5HT -> more exploration

    def apply_to_signal(self, x: np.ndarray,
                        noise_rng: np.random.Generator = None) -> np.ndarray:
        """Apply gain and noise suppression to a signal."""
        x = np.asarray(x, dtype=np.float64) * self.gain
        if self.noise_threshold > 0:
            x = np.where(np.abs(x) > self.noise_threshold, x, 0.0)
        return x


def neuromod_to_params(state: NeuromodulatorState) -> ModulatableParams:
    """Convert neuromodulator levels to effect parameters.

    Maps:
      DA  -> learning_rate_scale   (linear: 0->0.2x, 0.5->1.0x, 1->2.0x)
      NE  -> gain                  (linear: 0->0.5x, 0.5->1.0x, 1->2.0x)
      ACh -> noise_threshold       (linear: 0->0, 0.5->0, 1->0.1 of signal std)
      5HT -> discount_rate, exploration
    """
    lr_scale  = 0.2 + 1.8 * state.DA               # [0.2, 2.0]
    gain      = 0.5 + 1.5 * state.NE               # [0.5, 2.0]
    noise_thr = 0.1 * max(state.ACh - 0.5, 0) * 2  # [0, 0.1]
    gamma     = 0.80 + 0.19 * state.HT5            # [0.80, 0.99]
    eps       = 0.5 * (1.0 - state.HT5)            # [0, 0.5]
    return ModulatableParams(
        learning_rate_scale=float(lr_scale),
        gain=float(gain),
        noise_threshold=float(noise_thr),
        discount_rate=float(gamma),
        exploration=float(eps),
    )


# ---------------------------------------------------------------------------
# NeuromodulatorBroadcast  -- sends state to registered modules
# ---------------------------------------------------------------------------

class NeuromodulatorBroadcast:
    """Broadcasts neuromodulator state to all registered brain modules.

    Each registered module provides a callback that is called with
    a ModulatableParams object whenever the broadcast fires.

    Usage
    -----
        nms = NeuromodulatorSystem()
        broadcast = NeuromodulatorBroadcast(nms)

        # Register a module -- it receives params on every broadcast.
        broadcast.register("hippocampus", my_hippocampus_update_fn)

        # Fire events and broadcast.
        nms.reward_signal(rpe=+0.5)
        broadcast.fire()   # all registered modules receive updated params
    """

    def __init__(self, system: NeuromodulatorSystem) -> None:
        self.system    = system
        self._modules: dict[str, Callable[[ModulatableParams], None]] = {}
        self._history: list[NeuromodulatorState] = []

    def register(self, name: str,
                 callback: Callable[[ModulatableParams], None]) -> None:
        """Register a brain module to receive broadcast updates."""
        self._modules[name] = callback

    def unregister(self, name: str) -> None:
        self._modules.pop(name, None)

    def fire(self) -> ModulatableParams:
        """Translate current state to params and call all registered modules."""
        params = neuromod_to_params(self.system.state)
        for name, cb in self._modules.items():
            cb(params)
        self._history.append(NeuromodulatorState(
            DA=self.system.state.DA, ACh=self.system.state.ACh,
            NE=self.system.state.NE, HT5=self.system.state.HT5))
        return params

    def step_and_fire(self, dt: float = 1.0) -> ModulatableParams:
        """Decay neuromodulator levels then broadcast."""
        self.system.step(dt)
        return self.fire()

    def current_params(self) -> ModulatableParams:
        return neuromod_to_params(self.system.state)

    def history_array(self) -> np.ndarray:
        """Return history as (T, 4) array: [DA, ACh, NE, 5HT]."""
        if not self._history:
            return np.zeros((0, 4))
        return np.array([s.as_array() for s in self._history])

    def registered_modules(self) -> list[str]:
        return list(self._modules.keys())
