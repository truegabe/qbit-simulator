"""Quantum autoencoder tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.quantum_autoencoder import (
    trash_overlap, train_autoencoder, encode, compression_fidelity,
    _reduced_density_matrix, _build_ansatz_matrix,
)
from qbit_simulator.algorithms.ssvqe import hardware_efficient_ansatz_apply


# ---- Reduced density matrix ----

def test_reduced_density_matrix_full_keep():
    psi = np.array([1, 0, 0, 0], dtype=complex)
    rho = _reduced_density_matrix(psi, n_total=2, keep_qubits=[0, 1])
    # Full = |ψ⟩⟨ψ|.
    expected = np.outer(psi, psi.conj())
    assert np.allclose(rho, expected, atol=1e-12)


def test_reduced_density_matrix_trace_one():
    rng = np.random.default_rng(0)
    psi = rng.normal(size=8) + 1j * rng.normal(size=8)
    psi /= np.linalg.norm(psi)
    rho = _reduced_density_matrix(psi, n_total=3, keep_qubits=[0])
    assert abs(np.trace(rho).real - 1.0) < 1e-10


def test_reduced_density_matrix_bell_state():
    """Bell state |Φ+⟩ — each qubit reduces to I/2."""
    psi = np.zeros(4, dtype=complex)
    psi[0] = 1 / np.sqrt(2)
    psi[3] = 1 / np.sqrt(2)
    rho_0 = _reduced_density_matrix(psi, n_total=2, keep_qubits=[0])
    assert np.allclose(rho_0, np.eye(2) / 2, atol=1e-9)


# ---- Trash overlap ----

def test_trash_overlap_in_range():
    ansatz, n_p = hardware_efficient_ansatz_apply(n_qubits=3, depth=2)
    rng = np.random.default_rng(0)
    params = rng.uniform(-0.5, 0.5, size=n_p)
    psi = np.zeros(8, dtype=complex); psi[0] = 1.0
    overlap = trash_overlap(psi, params, ansatz, n_qubits=3, n_keep=2)
    assert 0.0 <= overlap <= 1.0


def test_trash_overlap_identity_no_compression():
    """At θ=0, ansatz is identity → overlap with |0⟩ trash = |⟨0_trash|ψ⟩|²."""
    ansatz, n_p = hardware_efficient_ansatz_apply(n_qubits=3, depth=2)
    # |000⟩ → all in |0⟩, overlap = 1.
    psi_zero = np.zeros(8, dtype=complex); psi_zero[0] = 1.0
    overlap = trash_overlap(psi_zero, np.zeros(n_p), ansatz, 3, 2)
    assert abs(overlap - 1.0) < 1e-9


# ---- Training ----

def test_training_returns_structure():
    rng = np.random.default_rng(0)
    psi = np.zeros(8, dtype=complex); psi[0] = 1.0
    result = train_autoencoder([psi], n_qubits=3, n_keep=2, depth=1,
                                 n_iter=3, lr=0.1, rng=rng)
    assert "params" in result
    assert "loss_history" in result
    assert "final_loss" in result


def test_training_decreases_loss():
    """Loss should generally decrease over training."""
    rng = np.random.default_rng(0)
    # Compressible: 1 qubit of redundancy.
    states = []
    for _ in range(3):
        psi_2q = rng.normal(size=4) + 1j * rng.normal(size=4)
        psi_2q /= np.linalg.norm(psi_2q)
        psi_3q = np.zeros(8, dtype=complex); psi_3q[:4] = psi_2q
        states.append(psi_3q)
    result = train_autoencoder(states, n_qubits=3, n_keep=2, depth=2,
                                 n_iter=20, lr=0.2, rng=rng)
    assert result["loss_history"][-1] <= result["loss_history"][0]


def test_training_loss_always_nonneg():
    rng = np.random.default_rng(0)
    psi = np.zeros(8, dtype=complex); psi[0] = 1.0
    result = train_autoencoder([psi], n_qubits=3, n_keep=2, depth=1,
                                 n_iter=5, rng=rng)
    for L in result["loss_history"]:
        assert L >= -1e-9


# ---- Compression fidelity ----

def test_compression_fidelity_in_range():
    ansatz, n_p = hardware_efficient_ansatz_apply(n_qubits=3, depth=2)
    params = np.zeros(n_p)
    psi = np.zeros(8, dtype=complex); psi[0] = 1.0
    f = compression_fidelity(psi, params, ansatz, n_qubits=3, n_keep=2)
    assert 0.0 <= f <= 1.0 + 1e-9


def test_compression_perfect_for_zero_state():
    """|000⟩ should reconstruct perfectly under identity encoder."""
    ansatz, n_p = hardware_efficient_ansatz_apply(n_qubits=3, depth=2)
    params = np.zeros(n_p)
    psi = np.zeros(8, dtype=complex); psi[0] = 1.0
    f = compression_fidelity(psi, params, ansatz, n_qubits=3, n_keep=2)
    assert f > 0.99


# ---- Ansatz matrix builder ----

def test_build_ansatz_matrix_is_unitary():
    ansatz, n_p = hardware_efficient_ansatz_apply(n_qubits=2, depth=2)
    rng = np.random.default_rng(0)
    params = rng.uniform(-1, 1, size=n_p)
    U = _build_ansatz_matrix(ansatz, params, 2)
    assert np.allclose(U @ U.conj().T, np.eye(4), atol=1e-9)


# ---- Encode wrapper ----

def test_encode_returns_normalized_state():
    ansatz, n_p = hardware_efficient_ansatz_apply(n_qubits=3, depth=2)
    rng = np.random.default_rng(0)
    params = rng.uniform(-1, 1, size=n_p)
    psi = np.zeros(8, dtype=complex); psi[0] = 1.0
    out = encode(psi, params, ansatz)
    assert abs(np.linalg.norm(out) - 1.0) < 1e-10
