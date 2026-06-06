"""Multi-qubit process tomography.

Extends the single-qubit process tomography in `tomography.py` to
arbitrary n-qubit channels.

For an n-qubit channel ε, the **Choi matrix** is

    J(ε) = sum_{i,j} |i⟩⟨j| ⊗ ε(|i⟩⟨j|)        (dim 2^(2n) × 2^(2n))

which fully characterizes ε. To reconstruct J empirically:

  1. Prepare 4^n tomographically-complete input states. A standard
     choice: product states from {|0⟩, |1⟩, |+⟩, |i+⟩}.
  2. Apply ε to each input.
  3. Perform state tomography on each output → 4^n Pauli expectation
     vectors, each of length 4^n.
  4. Solve the linear system to extract J(ε).

This module provides:

  - `tomography_input_states(n_qubits)`: the 4^n SIC-like product input
    states (as state vectors).
  - `multi_qubit_process_tomography(channel_fn, n_qubits)`: full Choi
    reconstruction.
  - `choi_to_kraus(choi)`: extract Kraus operators from a Choi matrix.
  - `process_fidelity(choi_real, choi_ideal)`: average gate fidelity.

We work in the dense matrix representation (good for n ≤ 4 qubits).
"""

from __future__ import annotations

from typing import Callable
from itertools import product

import numpy as np

from .tomography import (
    pauli_string_matrix, all_pauli_strings,
    exact_pauli_expectations, reconstruct_density_matrix,
)


# ----------------------------------------------------------------------------
# Single-qubit informationally-complete input set
# ----------------------------------------------------------------------------

_SINGLE_QUBIT_INPUTS = {
    "0":  np.array([1, 0], dtype=np.complex128),
    "1":  np.array([0, 1], dtype=np.complex128),
    "+":  np.array([1, 1], dtype=np.complex128) / np.sqrt(2),
    "i+": np.array([1, 1j], dtype=np.complex128) / np.sqrt(2),
}


def tomography_input_states(n_qubits: int) -> dict[str, np.ndarray]:
    """All 4^n product input states from {|0⟩, |1⟩, |+⟩, |i+⟩}^⊗n.

    Returns:
        dict mapping a string label (e.g. "0+i+") to its state vector.
    """
    labels = list(_SINGLE_QUBIT_INPUTS.keys())
    out = {}
    for combo in product(labels, repeat=n_qubits):
        psi = np.array([1.0 + 0j])
        for lab in combo:
            psi = np.kron(psi, _SINGLE_QUBIT_INPUTS[lab])
        # Use a short label like "0,+,1" for n>1; for n=1 just the label.
        key = ",".join(combo)
        out[key] = psi
    return out


# ----------------------------------------------------------------------------
# Channel application + tomography
# ----------------------------------------------------------------------------

def apply_channel(channel_fn: Callable[[np.ndarray], np.ndarray],
                   rho_in: np.ndarray) -> np.ndarray:
    """Wrapper: apply channel_fn to rho_in. The user-supplied callable
    can take either a state vector (pure input) or a density matrix.
    We always pass a density matrix."""
    rho_out = channel_fn(rho_in)
    if rho_out.shape != rho_in.shape:
        raise ValueError(f"channel changed dimension {rho_in.shape} → {rho_out.shape}")
    return rho_out


# ----------------------------------------------------------------------------
# Choi matrix reconstruction
# ----------------------------------------------------------------------------

def multi_qubit_process_tomography(
    channel_fn: Callable[[np.ndarray], np.ndarray],
    n_qubits: int,
) -> np.ndarray:
    """Reconstruct the Choi matrix J(ε) of an n-qubit channel ε.

    Uses the Pauli-basis approach:
        ε(P) = sum_Q  C[Q, P] · Q
    where C is the "Pauli transfer matrix" / process matrix. We compute
    C and convert to the Choi.

    Args:
        channel_fn:  callable rho → ε(rho), input/output 2^n × 2^n.
        n_qubits:    n.

    Returns:
        Choi matrix J(ε), shape (2^(2n), 2^(2n)).
    """
    d = 2 ** n_qubits
    pauli_strings = all_pauli_strings(n_qubits)
    n_paulis = len(pauli_strings)
    assert n_paulis == d ** 2

    # Pauli matrices (orthogonal basis for d×d Hermitian).
    P_matrices = [pauli_string_matrix(s) for s in pauli_strings]

    # Apply channel to each Pauli (channel is LINEAR, so we can do this
    # for an OPERATOR input by linearity in the input).
    # ε(P_k) is a d×d matrix.
    transfer_matrix = np.zeros((n_paulis, n_paulis), dtype=np.complex128)
    for k, P_in in enumerate(P_matrices):
        out = channel_fn(P_in)
        # Decompose out = sum_j c_jk P_j: c_jk = Tr(P_j out) / d.
        for j, P_out in enumerate(P_matrices):
            transfer_matrix[j, k] = np.trace(P_out @ out) / d

    # Now convert Pauli transfer matrix C → Choi J(ε).
    # Relation: J(ε) = sum_{j,k} C[j,k] · (P_k^T ⊗ P_j) / d.
    # (This is a standard identity; see Nielsen-Chuang or Wood et al.)
    J = np.zeros((d * d, d * d), dtype=np.complex128)
    for k in range(n_paulis):
        for j in range(n_paulis):
            J += transfer_matrix[j, k] * np.kron(P_matrices[k].T, P_matrices[j]) / d

    return J


# ----------------------------------------------------------------------------
# Choi → Kraus
# ----------------------------------------------------------------------------

def choi_to_kraus(choi: np.ndarray, threshold: float = 1e-10
                    ) -> list[np.ndarray]:
    """Extract Kraus operators from a Choi matrix via eigendecomposition.

    J(ε) = sum_k λ_k |v_k⟩⟨v_k|;  Kraus operators K_k = sqrt(λ_k)
    reshape(v_k) (in row-major form). Negative or near-zero eigenvalues
    are dropped (threshold).
    """
    d2 = choi.shape[0]
    d = int(np.sqrt(d2))
    assert d * d == d2

    eigvals, eigvecs = np.linalg.eigh(choi)
    kraus = []
    for k in range(d2):
        if eigvals[k] > threshold:
            v = eigvecs[:, k] * np.sqrt(eigvals[k])
            K = v.reshape(d, d)
            kraus.append(K)
    return kraus


def kraus_to_choi(kraus_ops: list[np.ndarray]) -> np.ndarray:
    """Build the Choi matrix from a Kraus decomposition.

    J(ε) = sum_k |K_k⟩⟩ ⟨⟨K_k|  where |K⟩⟩ = K.flatten() (row-major).
    """
    d = kraus_ops[0].shape[0]
    J = np.zeros((d * d, d * d), dtype=np.complex128)
    for K in kraus_ops:
        v = K.flatten()
        J += np.outer(v, v.conj())
    return J


# ----------------------------------------------------------------------------
# Process fidelity
# ----------------------------------------------------------------------------

def process_fidelity_choi(J_real: np.ndarray, J_ideal: np.ndarray) -> float:
    """Process fidelity between two channels via their Choi matrices.

        F_pro(ε_1, ε_2) = Tr(J_1 J_2) / d²

    where d is the system dimension. F=1 for identical channels.
    """
    d2 = J_real.shape[0]
    d = int(np.sqrt(d2))
    return float(np.real(np.trace(J_real @ J_ideal)) / d ** 2)


def average_gate_fidelity_channel(J_real: np.ndarray,
                                    U_ideal: np.ndarray) -> float:
    """Average gate fidelity between a channel (Choi) and a unitary target U.

        F_avg = (d · F_pro + 1) / (d + 1)
    where F_pro = Tr(J_real · J_U) / d² and J_U is the unitary Choi.
    """
    d = U_ideal.shape[0]
    # Unitary Choi: |U⟩⟩⟨⟨U| where |U⟩⟩ = U.flatten().
    u_flat = U_ideal.flatten()
    J_U = np.outer(u_flat, u_flat.conj())
    F_pro = process_fidelity_choi(J_real, J_U)
    return float((d * F_pro + 1) / (d + 1))
