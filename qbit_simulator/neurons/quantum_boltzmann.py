"""Quantum Boltzmann machine (toy).

A QBM replaces the classical Hamiltonian H_cl = -sum b_i s_i - sum J_ij s_i s_j
with a transverse-field Ising Hamiltonian:

    H = -sum_i (b_i Z_i + Γ_i X_i) - sum_{ij} J_ij Z_i Z_j

The Gibbs state ρ = exp(-β H) / Z encodes the QBM distribution over
classical configurations (measured in the Z basis).

For tiny systems (n ≤ 6), we build H exactly as a 2^n × 2^n matrix and
compute ρ via matrix exponentiation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# Single-qubit Paulis.
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def kron_op(op: np.ndarray, idx: int, n_qubits: int) -> np.ndarray:
    """Tensor-product `op` at position `idx` with identities elsewhere."""
    ops = [I2] * n_qubits
    ops[idx] = op
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return out


@dataclass
class QuantumBoltzmann:
    n_qubits: int
    b: np.ndarray = field(default=None, repr=False)
    Gamma: np.ndarray = field(default=None, repr=False)
    J: np.ndarray = field(default=None, repr=False)
    beta: float = 1.0

    def __post_init__(self) -> None:
        if self.b is None:
            self.b = np.zeros(self.n_qubits)
        if self.Gamma is None:
            self.Gamma = np.zeros(self.n_qubits)
        if self.J is None:
            self.J = np.zeros((self.n_qubits, self.n_qubits))

    def hamiltonian(self) -> np.ndarray:
        n = self.n_qubits
        d = 2 ** n
        H = np.zeros((d, d), dtype=complex)
        for i in range(n):
            H -= self.b[i] * kron_op(Z, i, n)
            H -= self.Gamma[i] * kron_op(X, i, n)
            for j in range(i + 1, n):
                if self.J[i, j] != 0:
                    H -= self.J[i, j] * (kron_op(Z, i, n) @ kron_op(Z, j, n))
        return H

    def density_matrix(self) -> np.ndarray:
        H = self.hamiltonian()
        evals, evecs = np.linalg.eigh(H)
        ev = np.exp(-self.beta * evals)
        Z = ev.sum()
        rho = evecs @ np.diag(ev / Z) @ evecs.conj().T
        return rho

    def marginals(self) -> np.ndarray:
        """Per-qubit ⟨Z_i⟩ under the QBM."""
        rho = self.density_matrix()
        out = np.zeros(self.n_qubits)
        for i in range(self.n_qubits):
            out[i] = float(np.real(np.trace(rho @ kron_op(Z, i, self.n_qubits))))
        return out

    def classical_probs(self) -> np.ndarray:
        """Probabilities over the 2^n classical states (Z basis)."""
        rho = self.density_matrix()
        return np.real(np.diag(rho))
