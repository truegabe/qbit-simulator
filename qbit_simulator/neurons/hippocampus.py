"""Hippocampus CA3/CA1 model: pattern separation + completion.

Two complementary modules:
  - DG (dentate gyrus): sparse encoding → pattern separation
    (similar inputs get orthogonalized into distinct codes).
  - CA3: dense recurrent network → pattern completion
    (a partial input is auto-associatively completed).
  - CA1: linear decoder from CA3 to output.

Encoding (storage):  CA3 weights learn outer-product Hebbian rule on
    sparse DG-encoded patterns.
Retrieval:           Iterate CA3 with WTA dynamics from a partial cue,
    decode through CA1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Hippocampus:
    n_input: int
    n_dg: int = 200
    n_ca3: int = 100
    sparsity: float = 0.05
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)
    W_in_dg: np.ndarray = field(default=None, repr=False)
    W_dg_ca3: np.ndarray = field(default=None, repr=False)
    W_ca3_ca3: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.W_in_dg is None:
            self.W_in_dg = self.rng.normal(0, 1.0/np.sqrt(self.n_input),
                                              (self.n_dg, self.n_input))
        if self.W_dg_ca3 is None:
            self.W_dg_ca3 = self.rng.normal(0, 1.0/np.sqrt(self.n_dg),
                                               (self.n_ca3, self.n_dg))
        if self.W_ca3_ca3 is None:
            self.W_ca3_ca3 = np.zeros((self.n_ca3, self.n_ca3))

    def _topk(self, x: np.ndarray, k: int) -> np.ndarray:
        if k >= len(x):
            return (x > x.mean()).astype(np.float64)
        thresh = np.partition(x, -k)[-k]
        return (x >= thresh).astype(np.float64)

    def encode_dg(self, x: np.ndarray) -> np.ndarray:
        k = max(int(self.sparsity * self.n_dg), 1)
        return self._topk(self.W_in_dg @ x, k)

    def encode_ca3(self, dg: np.ndarray) -> np.ndarray:
        k = max(int(self.sparsity * self.n_ca3), 1)
        return self._topk(self.W_dg_ca3 @ dg, k)

    def store(self, x: np.ndarray) -> None:
        """Store a pattern: imprint on CA3 recurrent matrix."""
        ca3 = self.encode_ca3(self.encode_dg(x))
        # Hebbian outer-product (Hopfield-like).
        self.W_ca3_ca3 += np.outer(ca3, ca3)
        np.fill_diagonal(self.W_ca3_ca3, 0)

    def retrieve(self, x: np.ndarray, n_iter: int = 5) -> np.ndarray:
        """Pattern completion: iterate CA3 dynamics from a cue."""
        ca3 = self.encode_ca3(self.encode_dg(x))
        k = max(int(self.sparsity * self.n_ca3), 1)
        for _ in range(n_iter):
            h = self.W_ca3_ca3 @ ca3
            ca3 = self._topk(h, k)
        return ca3

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        ca = self.retrieve(a, n_iter=3)
        cb = self.retrieve(b, n_iter=3)
        denom = (np.linalg.norm(ca) * np.linalg.norm(cb)) + 1e-9
        return float((ca @ cb) / denom)
