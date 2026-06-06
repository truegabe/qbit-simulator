"""Adaptive Resonance Theory (Grossberg / Carpenter).

ART-1 is an online clustering algorithm that solves the "stability-
plasticity dilemma": it can learn new patterns without forgetting old
ones, by creating new clusters when input does not resonate with any
existing prototype.

ART-1 algorithm (binary inputs):
  1. Choose the category J that maximizes T_J = |x ∧ w_J| / (β + |w_J|).
  2. Vigilance test: |x ∧ w_J| / |x| ≥ ρ?
       - Yes: resonance — update w_J <- x ∧ w_J.
       - No: try the next-best category, etc. If none, create a new one.

ρ ∈ [0, 1] is the vigilance parameter — higher ρ = finer categories.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ART1:
    """ART-1 network for binary input vectors."""
    input_dim: int
    rho: float = 0.5
    beta: float = 0.1
    max_categories: int = 64
    weights: list = field(default_factory=list)

    @property
    def n_categories(self) -> int:
        return len(self.weights)

    def _match(self, x: np.ndarray, w: np.ndarray) -> float:
        x_and_w = np.minimum(x, w)
        return float(x_and_w.sum() / (self.beta + w.sum()))

    def _vigilance_score(self, x: np.ndarray, w: np.ndarray) -> float:
        x_and_w = np.minimum(x, w)
        return float(x_and_w.sum() / max(x.sum(), 1e-12))

    def present(self, x: np.ndarray, learn: bool = True) -> int:
        """Present one binary input. Returns the chosen category index
        (or -1 if all are rejected and no slot is free).
        """
        x = (x > 0).astype(np.float64)
        # Rank categories by match T_J.
        T_scores = [self._match(x, w) for w in self.weights]
        order = sorted(range(len(self.weights)), key=lambda i: -T_scores[i])
        for J in order:
            if self._vigilance_score(x, self.weights[J]) >= self.rho:
                # Resonance.
                if learn:
                    self.weights[J] = np.minimum(x, self.weights[J])
                return J
        # No category accepted — create a new one if possible.
        if len(self.weights) < self.max_categories:
            self.weights.append(x.copy())
            return len(self.weights) - 1
        return -1

    def predict(self, x: np.ndarray) -> int:
        return self.present(x, learn=False)
