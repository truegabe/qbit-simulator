"""Quantum reservoir computing.

A small fixed quantum system, driven by classical inputs, acts as a
high-dimensional nonlinear reservoir. The reservoir state is read out
by measuring Pauli expectation values; a linear readout layer is
trained classically.

Setup:
  - n_qubits qubits initialized to a fixed state.
  - At each time step t:
      1. Apply input-dependent unitary U(x_t).
      2. Apply fixed random "scrambling" unitary V.
      3. Read out ⟨Z_i⟩ for i = 1, ..., n_qubits — the reservoir features.
  - Train a linear map from these features to the target output.

Universal-approximation property: large enough quantum reservoirs can
approximate any input/output function.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .quantum_boltzmann import kron_op, X, Z, I2


def random_unitary(d: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    Q, R = np.linalg.qr(A)
    return Q * (np.diag(R) / np.abs(np.diag(R)))[None, :]


@dataclass
class QuantumReservoir:
    n_qubits: int = 4
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)
    V: np.ndarray = field(default=None, repr=False)
    psi: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        d = 2 ** self.n_qubits
        if self.V is None:
            self.V = random_unitary(d, self.rng)
        if self.psi is None:
            self.psi = np.zeros(d, dtype=complex)
            self.psi[0] = 1.0

    def input_unitary(self, x: np.ndarray) -> np.ndarray:
        """Encode scalar / vector input as rotations on each qubit."""
        x = np.atleast_1d(x)
        n = self.n_qubits
        d = 2 ** n
        U = np.eye(d, dtype=complex)
        for q in range(n):
            ang = x[q % len(x)] * np.pi
            R = np.array([[np.cos(ang/2), -1j*np.sin(ang/2)],
                           [-1j*np.sin(ang/2), np.cos(ang/2)]], dtype=complex)
            U = kron_op(R, q, n) @ U
        return U

    def step(self, x: np.ndarray) -> np.ndarray:
        """One time step. Returns ⟨Z_i⟩ for all qubits."""
        U = self.input_unitary(x)
        self.psi = self.V @ U @ self.psi
        self.psi /= np.linalg.norm(self.psi)
        # Measure ⟨Z_i⟩.
        out = np.zeros(self.n_qubits)
        n = self.n_qubits
        for i in range(n):
            Zi = kron_op(Z, i, n)
            out[i] = float(np.real(self.psi.conj() @ Zi @ self.psi))
        return out

    def run(self, X: np.ndarray) -> np.ndarray:
        feats = []
        for x in X:
            feats.append(self.step(x))
        return np.array(feats)

    def reset(self) -> None:
        self.psi = np.zeros(2 ** self.n_qubits, dtype=complex)
        self.psi[0] = 1.0


def train_linear_readout(features: np.ndarray, targets: np.ndarray,
                          reg: float = 1e-3) -> np.ndarray:
    """Ridge regression readout."""
    n_feat = features.shape[1]
    A = features.T @ features + reg * np.eye(n_feat)
    return np.linalg.solve(A, features.T @ targets)
