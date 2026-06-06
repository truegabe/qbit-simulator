"""Prefrontal cortex working memory model.

PFC maintains task-relevant info across delays via a gated attractor:
  - "store" signal admits new content,
  - "ignore" signal blocks distractors,
  - sustained persistent activity holds the content.

Model: each "slot" is a bistable unit with input gating. Slots share
mutual inhibition (winner-take-all over slots).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PFCWorkingMemory:
    n_slots: int = 4
    n_features: int = 8
    threshold: float = 0.5
    decay: float = 0.02
    lateral_inhibition: float = 0.3
    contents: np.ndarray = field(default=None, repr=False)
    occupancy: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.contents is None:
            self.contents = np.zeros((self.n_slots, self.n_features))
        if self.occupancy is None:
            self.occupancy = np.zeros(self.n_slots)

    def store(self, x: np.ndarray, gate: float = 1.0) -> int:
        """Store pattern x in the first free slot if gate>threshold.

        Returns the slot index used (or -1 if all full / gate closed).
        """
        if gate < self.threshold:
            return -1
        for i in range(self.n_slots):
            if self.occupancy[i] < self.threshold:
                self.contents[i] = x
                self.occupancy[i] = 1.0
                return i
        return -1   # all full

    def step(self) -> None:
        """One step of attractor dynamics: persistent + slow decay + lateral inhibition."""
        # Slow leak.
        self.occupancy -= self.decay
        # Lateral inhibition: each slot's occupancy is reduced by sum of others.
        total = self.occupancy.sum()
        self.occupancy -= self.lateral_inhibition * (total - self.occupancy) * 0.01
        self.occupancy = np.clip(self.occupancy, 0, 1)

    def read(self, slot: int) -> np.ndarray | None:
        if self.occupancy[slot] >= self.threshold:
            return self.contents[slot].copy()
        return None

    def clear(self, slot: int = -1) -> None:
        if slot < 0:
            self.contents[:] = 0
            self.occupancy[:] = 0
        else:
            self.contents[slot] = 0
            self.occupancy[slot] = 0
