"""Neural Turing Machine (Graves et al. 2014) — simplified.

A controller network is augmented with an external memory bank
M ∈ R^{N×D} and differentiable read/write heads:

    Read:  r_t = sum_i w_t(i) M_t(i)
    Write: M_{t+1}(i) = M_t(i) (1 - w_t(i) e_t) + w_t(i) a_t

where w_t is a soft attention vector over memory rows, e_t the erase
vector and a_t the add vector. The controller produces the head
addresses by a combination of content-based and location-based
addressing.

We implement a simplified version: content-based addressing only,
linear controller, suitable for tasks like associative recall.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


@dataclass
class NeuralTuringMachine:
    """Minimal NTM with content-based addressing.

    Memory: M of shape (N, D).
    """
    N: int = 32          # number of memory rows
    D: int = 16          # row width
    n_in: int = 8        # controller input dim
    n_out: int = 8       # controller output dim
    M: np.ndarray = field(default=None, repr=False)
    W_read: np.ndarray = field(default=None, repr=False)
    W_write: np.ndarray = field(default=None, repr=False)
    W_erase: np.ndarray = field(default=None, repr=False)
    W_add: np.ndarray = field(default=None, repr=False)
    W_out: np.ndarray = field(default=None, repr=False)
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    def __post_init__(self) -> None:
        if self.M is None:
            self.M = np.ones((self.N, self.D)) * 1e-3
        std = 0.1
        if self.W_read is None:
            self.W_read = self.rng.normal(0, std, (self.D, self.n_in))
        if self.W_write is None:
            self.W_write = self.rng.normal(0, std, (self.D, self.n_in))
        if self.W_erase is None:
            self.W_erase = self.rng.normal(0, std, (self.D, self.n_in))
        if self.W_add is None:
            self.W_add = self.rng.normal(0, std, (self.D, self.n_in))
        if self.W_out is None:
            self.W_out = self.rng.normal(0, std, (self.n_out, self.D + self.n_in))

    def _address(self, key: np.ndarray, beta: float = 1.0) -> np.ndarray:
        """Cosine-similarity content addressing."""
        K = self.M / (np.linalg.norm(self.M, axis=1, keepdims=True) + 1e-9)
        kn = key / (np.linalg.norm(key) + 1e-9)
        sim = K @ kn
        return softmax(beta * sim)

    def reset_memory(self) -> None:
        self.M = np.ones((self.N, self.D)) * 1e-3

    def step(self, x: np.ndarray, beta: float = 4.0) -> np.ndarray:
        """One time step. Returns the n_out output."""
        # Heads.
        k_r = self.W_read  @ x
        k_w = self.W_write @ x
        e   = 1.0 / (1.0 + np.exp(-(self.W_erase @ x)))   # sigmoid in (0,1)
        a   = self.W_add   @ x
        # Read.
        w_r = self._address(k_r, beta=beta)
        r = w_r @ self.M
        # Write.
        w_w = self._address(k_w, beta=beta)
        self.M = self.M * (1 - np.outer(w_w, e)) + np.outer(w_w, a)
        # Output.
        z = np.concatenate([r, x])
        return self.W_out @ z

    def write_at(self, idx: int, value: np.ndarray) -> None:
        """Direct write for setting up associative-recall tasks."""
        self.M[idx % self.N] = value

    def read_at(self, idx: int) -> np.ndarray:
        return self.M[idx % self.N].copy()
