"""Density matrix representation for mixed states.

For N qubits, a density matrix ρ is a 2^N × 2^N positive semidefinite
Hermitian matrix with trace 1. This format is needed to represent:
  - Statistical mixtures (e.g. uniform mixture of |0⟩ and |1⟩ is ρ = I/2)
  - Subsystems of entangled states (via partial trace)
  - Open quantum systems under noise (exact, not Monte Carlo)

Costs: memory is 4^N (vs 2^N for state vectors). Practical ceiling ~12 qubits
on this laptop. Pure-state simulation should keep using `QuantumCircuit`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


class DensityMatrix:
    def __init__(self, rho: np.ndarray):
        rho = np.asarray(rho, dtype=np.complex128)
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
            raise ValueError(f"rho must be square 2D, got shape {rho.shape}")
        dim = rho.shape[0]
        n = int(np.log2(dim))
        if 2**n != dim:
            raise ValueError(f"dimension {dim} is not a power of 2")
        self.rho = rho
        self.n = n

    # ---- factories ----

    @classmethod
    def from_state(cls, state: np.ndarray) -> "DensityMatrix":
        state = np.asarray(state, dtype=np.complex128)
        return cls(np.outer(state, state.conj()))

    @classmethod
    def mixed(cls, states: Sequence[np.ndarray], probs: Sequence[float]) -> "DensityMatrix":
        probs = np.asarray(probs, dtype=np.float64)
        if not np.isclose(probs.sum(), 1.0):
            raise ValueError("probabilities must sum to 1")
        rho = sum(p * np.outer(s, np.asarray(s).conj()) for p, s in zip(probs, states))
        return cls(rho)

    @classmethod
    def maximally_mixed(cls, n: int) -> "DensityMatrix":
        dim = 2**n
        return cls(np.eye(dim, dtype=np.complex128) / dim)

    # ---- operations ----

    def apply_unitary(self, U: np.ndarray, targets: Sequence[int]) -> "DensityMatrix":
        """ρ -> U ρ U†, where U acts on `targets` (others are identity)."""
        U_full = _embed_unitary(U, targets, self.n)
        self.rho = U_full @ self.rho @ U_full.conj().T
        return self

    def apply_kraus(self, kraus_ops: Sequence[np.ndarray], target: int) -> "DensityMatrix":
        """ρ -> Σ_i K_i ρ K_i† (exact channel application, no sampling)."""
        new_rho = np.zeros_like(self.rho)
        for K in kraus_ops:
            K_full = _embed_unitary(K, [target], self.n)
            new_rho += K_full @ self.rho @ K_full.conj().T
        self.rho = new_rho
        return self

    def partial_trace(self, qubits_to_trace: Sequence[int]) -> "DensityMatrix":
        """Trace out the listed qubits, returning a reduced density matrix."""
        keep = [q for q in range(self.n) if q not in qubits_to_trace]
        n_keep = len(keep)
        # Reshape ρ into (2,2,...,2,2,...) with N "row" + N "col" axes.
        shape = (2,) * (2 * self.n)
        T = self.rho.reshape(shape)
        # Trace over each qubit in qubits_to_trace.
        for q in sorted(qubits_to_trace, reverse=True):
            T = np.trace(T, axis1=q, axis2=q + self.n - sum(1 for x in qubits_to_trace if x > q))
        # After all traces, T has shape (2,)*2*n_keep
        T = T.reshape(2**n_keep, 2**n_keep)
        return DensityMatrix(T)

    def expectation(self, observable: np.ndarray) -> float:
        """⟨O⟩ = Tr(ρ O)."""
        return float(np.real(np.trace(self.rho @ observable)))

    def probabilities(self) -> np.ndarray:
        """Diagonal of ρ — probabilities of each computational basis outcome."""
        return np.real(np.diag(self.rho))

    def purity(self) -> float:
        """Tr(ρ²): 1 for pure states, 1/dim for maximally mixed."""
        return float(np.real(np.trace(self.rho @ self.rho)))

    def von_neumann_entropy(self) -> float:
        eigvals = np.linalg.eigvalsh(self.rho)
        eigvals = eigvals[eigvals > 1e-12]
        return float(-(eigvals * np.log2(eigvals)).sum())

    def __repr__(self) -> str:
        return f"DensityMatrix(n={self.n}, purity={self.purity():.4f})"


def _embed_unitary(U: np.ndarray, targets: Sequence[int], n_total: int) -> np.ndarray:
    """Build the 2^n_total operator that applies U to `targets` and I elsewhere."""
    k = len(targets)
    if U.shape != (2**k, 2**k):
        raise ValueError(f"U shape {U.shape} doesn't match {k} targets")
    # Use the same moveaxis trick as QuantumCircuit, applied to an identity.
    dim = 2**n_total
    full = np.eye(dim, dtype=np.complex128)
    tensor = full.reshape((2,) * (2 * n_total))
    # Treat the left half as "rows", apply U on target row axes.
    axes_in = list(targets)
    moved = np.moveaxis(tensor, axes_in, list(range(k)))
    shape = moved.shape
    moved = moved.reshape(2**k, -1)
    moved = U @ moved
    moved = moved.reshape(shape)
    moved = np.moveaxis(moved, list(range(k)), axes_in)
    return moved.reshape(dim, dim)
