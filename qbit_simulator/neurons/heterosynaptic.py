"""Heterosynaptic plasticity.

When one synapse undergoes LTP/LTD, neighboring synapses on the same
post-synaptic neuron get small COMPENSATORY adjustments. Two
biological motivations:
  - Synaptic-tagging-and-capture: a "tagged" synapse pulls plasticity-
    related proteins; nearby synapses share the pool, getting a
    smaller secondary boost.
  - Total-synaptic-weight homeostasis: prevents runaway potentiation
    by redistributing weight.

Update rule:
    Δw_target = η · pre_target · post · gating
    Δw_neighbor = -α · |Δw_target| · sign(w_neighbor)   for k != target

α controls the spread fraction (0 = pure Hebbian; 1 = fully
homeostatic).

This module operates on a weight VECTOR (one post-synaptic neuron with
many pre-synaptic inputs). Combine with the existing STDP / R-STDP /
three-factor modules: pass their update as the "primary" weight change
and `apply_heterosynaptic` redistributes it to neighbors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class HeterosynapticModulator:
    """Redistribute primary weight changes to neighboring synapses."""
    spread: float = 0.05     # fraction of |Δw| redistributed
    radius: int = 3          # neighborhood radius (0 = all neighbors)
    keep_total: bool = True  # enforce sum of weight changes ≈ 0

    def apply(self, dw_primary: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Take a vector of primary weight changes dw_primary; return the
        full update including heterosynaptic compensation."""
        n = len(w)
        dw_total = dw_primary.copy().astype(np.float64)
        for i in range(n):
            if abs(dw_primary[i]) < 1e-12:
                continue
            # Neighbors: within radius (or all if radius=0).
            if self.radius == 0:
                neigh = [j for j in range(n) if j != i]
            else:
                neigh = [j for j in range(max(0, i - self.radius),
                                            min(n, i + self.radius + 1))
                          if j != i]
            if not neigh:
                continue
            # Each neighbor gets -spread * |dw_i| * sign(w_neighbor) / k.
            magnitude = self.spread * abs(dw_primary[i]) / len(neigh)
            for j in neigh:
                dw_total[j] -= magnitude * np.sign(w[j])
        if self.keep_total:
            # Enforce total weight change ≈ 0 by subtracting the residual
            # evenly from synapses other than the maximally-changed one.
            net = dw_total.sum()
            n_other = n - 1
            if n_other > 0:
                # Distribute residual to all synapses proportionally.
                dw_total -= net / n
        return dw_total


@dataclass
class HeterosynapticHebbian:
    """A simple Hebbian learner with built-in heterosynaptic redistribution."""
    n_inputs: int
    eta: float = 0.01
    modulator: HeterosynapticModulator = field(default=None)
    w: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.modulator is None:
            self.modulator = HeterosynapticModulator()
        if self.w is None:
            self.w = np.zeros(self.n_inputs)

    def step(self, x: np.ndarray, y: float) -> None:
        # Primary update: Hebbian.
        dw = self.eta * y * x
        dw = self.modulator.apply(dw, self.w)
        self.w += dw
