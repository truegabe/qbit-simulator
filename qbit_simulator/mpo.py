"""Matrix Product Operator (MPO) — 1D chain representation of a Hamiltonian.

Each site's operator is a rank-4 tensor of shape
    (chi_left, phys_out, phys_in, chi_right)

When contracted across all sites and traced over auxiliary bonds, this
reconstructs the full 2^N x 2^N Hamiltonian. For local Hamiltonians the
bond dimension stays small (3 for TFIM, 5 for Heisenberg), so the full
description fits in O(N) memory.

Standard builders are provided for two textbook 1D models:
    - Transverse-field Ising (TFIM): H = -J Σ Z_i Z_{i+1} - h Σ X_i
    - Heisenberg XXX:                H = J Σ (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})

Both have analytically known thermodynamic-limit ground energies; both are
the standard benchmarks for DMRG correctness.
"""

from __future__ import annotations

import numpy as np

from .gates import X, Y, Z, I2


class MPO:
    """Matrix Product Operator. `tensors[q]` has shape (l, p_out, p_in, r)."""

    def __init__(self, tensors: list[np.ndarray]):
        self.tensors = [t.astype(np.complex128) for t in tensors]
        self.n = len(tensors)
        # Sanity: chain bonds must be consistent and boundaries dim 1.
        if self.tensors[0].shape[0] != 1:
            raise ValueError("first MPO tensor must have left bond dim 1")
        if self.tensors[-1].shape[3] != 1:
            raise ValueError("last MPO tensor must have right bond dim 1")
        for q in range(self.n - 1):
            if self.tensors[q].shape[3] != self.tensors[q + 1].shape[0]:
                raise ValueError(f"MPO bond mismatch at site {q}/{q+1}")

    def bond_dimensions(self) -> list[int]:
        return [self.tensors[q].shape[3] for q in range(self.n - 1)]

    def to_dense(self) -> np.ndarray:
        """Full 2^N × 2^N matrix. Use only for small N."""
        # Contract the chain into a single tensor with all physical legs.
        W = self.tensors[0]                                # (1, p, p, r)
        for q in range(1, self.n):
            # W has shape (1, p1, p1, ..., r). Contract r with next tensor's l.
            W = np.tensordot(W, self.tensors[q], axes=([W.ndim - 1], [0]))
        # W shape: (1, p_out_0, p_in_0, p_out_1, p_in_1, ..., 1).
        # We want a 2^N × 2^N matrix indexed by (all p_outs, all p_ins).
        W = W.squeeze(axis=(0, -1))
        # Reorder axes: bring all p_out first then all p_in.
        # Current order: p_out_0, p_in_0, p_out_1, p_in_1, ...
        n = self.n
        outs = list(range(0, 2 * n, 2))
        ins  = list(range(1, 2 * n, 2))
        W = W.transpose(outs + ins)
        return W.reshape(2**n, 2**n)


# ---- builders for standard Hamiltonians ----

def _interior_tfim(J: float, h: float) -> np.ndarray:
    """TFIM interior MPO tensor with bond dim 3.

    W as a (3, 2, 2, 3) tensor:
        W[a, p_out, p_in, b] = matrix element of the a-th row b-th column.
    Layout (rows = a, cols = b):
            [ I        0        0   ]
            [ Z        0        0   ]
            [-h X    -J Z       I   ]
    """
    W = np.zeros((3, 2, 2, 3), dtype=np.complex128)
    W[0, :, :, 0] = I2
    W[1, :, :, 0] = Z
    W[2, :, :, 0] = -h * X
    W[2, :, :, 1] = -J * Z
    W[2, :, :, 2] = I2
    return W


def tfim_mpo(n: int, J: float = 1.0, h: float = 1.0) -> MPO:
    """Transverse-field Ising MPO: H = -J Σ Z_i Z_{i+1} - h Σ X_i."""
    interior = _interior_tfim(J, h)
    # Left boundary: take the last row of `interior` -> shape (1, 2, 2, 3).
    left = interior[2:3, :, :, :].copy()
    # Right boundary: take the first column -> shape (3, 2, 2, 1).
    right = interior[:, :, :, 0:1].copy()
    if n == 1:
        # Single-site special case: only the on-site -h X term.
        single = np.zeros((1, 2, 2, 1), dtype=np.complex128)
        single[0, :, :, 0] = -h * X
        return MPO([single])
    tensors = [left] + [interior.copy() for _ in range(n - 2)] + [right]
    return MPO(tensors)


def _interior_heisenberg(J: float) -> np.ndarray:
    """Heisenberg XXX interior MPO tensor with bond dim 5.

    W layout (5x5 block of 2x2 operators):
            [ I    0    0    0    0 ]
            [ X    0    0    0    0 ]
            [ Y    0    0    0    0 ]
            [ Z    0    0    0    0 ]
            [ 0   JX   JY   JZ    I ]
    """
    W = np.zeros((5, 2, 2, 5), dtype=np.complex128)
    W[0, :, :, 0] = I2
    W[1, :, :, 0] = X
    W[2, :, :, 0] = Y
    W[3, :, :, 0] = Z
    W[4, :, :, 1] = J * X
    W[4, :, :, 2] = J * Y
    W[4, :, :, 3] = J * Z
    W[4, :, :, 4] = I2
    return W


def heisenberg_mpo(n: int, J: float = 1.0) -> MPO:
    """Heisenberg XXX MPO: H = J Σ (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})."""
    interior = _interior_heisenberg(J)
    left = interior[4:5, :, :, :].copy()
    right = interior[:, :, :, 0:1].copy()
    if n == 1:
        single = np.zeros((1, 2, 2, 1), dtype=np.complex128)
        return MPO([single])
    tensors = [left] + [interior.copy() for _ in range(n - 2)] + [right]
    return MPO(tensors)
