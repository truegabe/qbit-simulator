"""Quantum perceptron + quantum Hopfield (toy versions).

Hybrid models that encode classical bit patterns as quantum states
and use unitary evolution for inference.

Quantum perceptron (Schuld et al.):
  - Encode input x ∈ {−1, +1}^n as |x⟩.
  - Apply parameterized unitary U(θ).
  - Measure output bit; train θ to match target.

Quantum Hopfield: stored patterns become a Hamiltonian whose ground
state is the closest stored pattern to a query.

Implemented over a small Hilbert space (2^n) with numpy linear algebra.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def basis_state(bits: np.ndarray) -> np.ndarray:
    """Encode bitstring as a Hilbert-space basis state."""
    n = len(bits)
    idx = int("".join(["1" if b > 0 else "0" for b in bits]), 2)
    psi = np.zeros(2 ** n, dtype=complex)
    psi[idx] = 1.0
    return psi


def apply_unitary(psi: np.ndarray, U: np.ndarray) -> np.ndarray:
    return U @ psi


def random_unitary(n: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random n×n unitary via QR decomposition."""
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    Q, R = np.linalg.qr(A)
    d = np.diag(R)
    Q = Q * (d / np.abs(d))[None, :]
    return Q


@dataclass
class QuantumHopfield:
    """Patterns become a projector Hamiltonian H = -sum_p |p><p|.

    Ground state of H is the stored pattern that has maximal overlap
    with the query.
    """
    patterns: list = field(default_factory=list)
    n_qubits: int = 4

    @property
    def H(self) -> np.ndarray:
        d = 2 ** self.n_qubits
        H = np.zeros((d, d), dtype=complex)
        for p in self.patterns:
            psi = basis_state(p)
            H -= np.outer(psi, psi.conj())
        return H

    def retrieve(self, query_bits: np.ndarray, beta: float = 5.0
                  ) -> np.ndarray:
        """Apply exp(-beta H) to the query and return the closest pattern."""
        H_eff = self.H
        # exp(-β H) via eigendecomposition.
        evals, evecs = np.linalg.eigh(H_eff)
        U = evecs @ np.diag(np.exp(-beta * evals)) @ evecs.conj().T
        psi = U @ basis_state(query_bits)
        psi /= np.linalg.norm(psi)
        # Decode by largest amplitude basis state.
        idx = int(np.argmax(np.abs(psi) ** 2))
        bits = np.array([(idx >> (self.n_qubits - 1 - k)) & 1
                          for k in range(self.n_qubits)])
        return np.where(bits > 0, 1, -1)


@dataclass
class QuantumPerceptron:
    """Variational classifier with a single output qubit.

    A parameterized unitary U(θ) is applied to a state-encoded input;
    Z-measurement on the first qubit gives the prediction.
    """
    n_qubits: int
    theta: np.ndarray = field(default=None, repr=False)
    eta: float = 0.05
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    def __post_init__(self) -> None:
        if self.theta is None:
            # 3 angles per qubit: simple single-qubit rotations.
            self.theta = self.rng.uniform(0, 2 * np.pi, size=3 * self.n_qubits)

    def _unitary(self) -> np.ndarray:
        d = 2 ** self.n_qubits
        U = np.eye(d, dtype=complex)
        # Single-qubit rotations Y-Z-Y on each qubit.
        for q in range(self.n_qubits):
            for axis_idx in range(3):
                ang = self.theta[3 * q + axis_idx]
                # Build rotation gate on qubit q.
                I = np.eye(2, dtype=complex)
                if axis_idx == 0 or axis_idx == 2:
                    R = np.array([[np.cos(ang/2), -np.sin(ang/2)],
                                   [np.sin(ang/2),  np.cos(ang/2)]], dtype=complex)
                else:
                    R = np.array([[np.exp(-1j*ang/2), 0],
                                   [0, np.exp(1j*ang/2)]], dtype=complex)
                # Kron up to full Hilbert space.
                ops = [I] * self.n_qubits
                ops[q] = R
                full = ops[0]
                for o in ops[1:]:
                    full = np.kron(full, o)
                U = full @ U
        return U

    def predict(self, x: np.ndarray) -> float:
        """Returns ⟨Z⟩ on the first qubit ∈ [-1, +1]."""
        psi = self._unitary() @ basis_state(x)
        # ⟨Z_0⟩ = sum_k (-1)^{first_bit(k)} |psi_k|^2
        d = 2 ** self.n_qubits
        signs = np.array([1 if (k >> (self.n_qubits - 1)) == 0 else -1
                          for k in range(d)])
        return float(np.real((np.abs(psi) ** 2) @ signs))
