"""Amygdala — fear conditioning (Pavlovian).

A CS (conditioned stimulus, e.g. tone) presented with a US
(unconditioned stimulus, e.g. shock) acquires fear-eliciting power.
Amygdala basolateral nucleus (BLA) is the substrate.

Simple model: a single fear-prediction unit that learns
   V(CS) <- V(CS) + α [US - V(CS)]
(Rescorla-Wagner). Extinction (CS without US) drives V back down but
context-gated re-acquisition is faster — a hallmark of fear memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Amygdala:
    n_stimuli: int
    alpha: float = 0.2
    V: np.ndarray = field(default=None, repr=False)
    extinction_V: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.V is None:
            self.V = np.zeros(self.n_stimuli)
        if self.extinction_V is None:
            self.extinction_V = np.zeros(self.n_stimuli)

    def fear_response(self, cs: np.ndarray, context: float = 1.0) -> float:
        """Aggregate fear response.

        Default (context=1, the extinction context): the OPPOSING
        extinction_V trace cancels acquired fear → low fear response.
        In a NEW context (context=0): extinction trace is gated off,
        so original fear renews — classic ABA renewal effect.
        """
        eff = np.maximum(self.V - self.extinction_V * context, 0)
        return float(eff @ cs)

    def trial(self, cs: np.ndarray, us: float, context: float = 1.0) -> float:
        """One conditioning / extinction trial.

        us = 1: acquisition trial (CS + shock).
        us = 0: extinction trial (CS only).
        Returns the prediction error.
        """
        # Current effective association = V minus context-gated extinction.
        V_pred = float((self.V - self.extinction_V * context) @ cs)
        delta = us - V_pred
        if us > 0:
            # Acquisition: standard Rescorla-Wagner on V.
            self.V += self.alpha * delta * cs
        else:
            # Extinction in this context: grow opposing trace.
            self.extinction_V += self.alpha * (-delta) * cs * context
        return float(delta)
