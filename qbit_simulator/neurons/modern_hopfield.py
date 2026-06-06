"""Modern Hopfield network (Ramsauer et al. 2020).

Continuous-state Hopfield with exponential interaction has
EXPONENTIAL storage capacity in N (compared to classical 0.14 N):

    Energy:    E(ξ) = -lse(β X^T ξ) + 0.5 ξ^T ξ + ...
    Update:    ξ <- X · softmax(β X^T ξ)

This is mathematically equivalent to softmax attention. So modern
Hopfield = attention in disguise — and it explains why transformers
work so well as associative memories.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


@dataclass
class ModernHopfield:
    """Stores N patterns of dim d. Each retrieval is one softmax-attention step."""
    patterns: np.ndarray
    beta: float = 1.0

    @property
    def N(self) -> int:
        return self.patterns.shape[0]

    @property
    def d(self) -> int:
        return self.patterns.shape[1]

    def retrieve(self, query: np.ndarray, n_steps: int = 1) -> np.ndarray:
        """One-shot attention is enough for well-separated patterns.

        For ambiguous queries, multiple steps approach a deeper minimum.
        """
        xi = query.copy().astype(np.float64)
        for _ in range(n_steps):
            scores = self.beta * self.patterns @ xi
            w = softmax(scores)
            xi = self.patterns.T @ w
        return xi

    def energy(self, xi: np.ndarray) -> float:
        scores = self.beta * self.patterns @ xi
        lse = np.log(np.exp(scores - scores.max()).sum()) + scores.max()
        return -lse / self.beta + 0.5 * (xi @ xi)

    def add_pattern(self, p: np.ndarray) -> None:
        self.patterns = np.vstack([self.patterns, p[None, :]])
