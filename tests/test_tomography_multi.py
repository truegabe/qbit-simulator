"""Multi-qubit process tomography tests."""

import numpy as np
import pytest

from qbit_simulator.tomography_multi import (
    tomography_input_states,
    multi_qubit_process_tomography,
    choi_to_kraus, kraus_to_choi,
    process_fidelity_choi, average_gate_fidelity_channel,
)


# ---- Input states ----

def test_input_states_count_single_qubit():
    states = tomography_input_states(1)
    assert len(states) == 4


def test_input_states_count_two_qubits():
    states = tomography_input_states(2)
    assert len(states) == 16


def test_input_states_are_normalized():
    for n in (1, 2):
        for label, psi in tomography_input_states(n).items():
            assert abs(np.linalg.norm(psi) - 1.0) < 1e-12


# ---- Identity channel ----

def test_identity_channel_choi_trace():
    """Choi of identity channel has trace d."""
    def identity(rho):
        return rho.copy()
    for n in (1, 2):
        d = 2 ** n
        J = multi_qubit_process_tomography(identity, n_qubits=n)
        assert abs(np.trace(J).real - d) < 1e-9


def test_identity_channel_max_eigenvalue():
    """Identity Choi = d · |Φ+⟩⟨Φ+|, one eigenvalue equals d."""
    def identity(rho):
        return rho.copy()
    J = multi_qubit_process_tomography(identity, n_qubits=2)
    eigs = np.linalg.eigvalsh(J)
    assert abs(eigs[-1] - 4.0) < 1e-9


# ---- Unitary channels ----

def test_cnot_channel_fidelity():
    """The reconstructed CNOT channel should match perfectly."""
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=complex)
    def cnot_channel(rho):
        return CNOT @ rho @ CNOT.conj().T
    J = multi_qubit_process_tomography(cnot_channel, n_qubits=2)
    F = average_gate_fidelity_channel(J, CNOT)
    assert abs(F - 1.0) < 1e-9


def test_cnot_choi_has_rank_one():
    """A unitary channel's Choi has rank 1."""
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=complex)
    def cnot_channel(rho):
        return CNOT @ rho @ CNOT.conj().T
    J = multi_qubit_process_tomography(cnot_channel, n_qubits=2)
    kraus = choi_to_kraus(J)
    assert len(kraus) == 1


def test_x_gate_channel_single_qubit():
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    def x_channel(rho):
        return X @ rho @ X
    J = multi_qubit_process_tomography(x_channel, n_qubits=1)
    F = average_gate_fidelity_channel(J, X)
    assert abs(F - 1.0) < 1e-9


# ---- Choi ↔ Kraus round trip ----

def test_kraus_to_choi_roundtrip():
    """Build Choi from Kraus, extract Kraus, rebuild Choi → match."""
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=complex)
    J1 = kraus_to_choi([CNOT])
    kraus = choi_to_kraus(J1)
    J2 = kraus_to_choi(kraus)
    assert np.allclose(J1, J2, atol=1e-9)


def test_kraus_decomposition_completeness():
    """sum_k K_k† K_k should equal identity for a trace-preserving channel."""
    def identity(rho):
        return rho
    J = multi_qubit_process_tomography(identity, n_qubits=2)
    kraus = choi_to_kraus(J)
    total = sum(K.conj().T @ K for K in kraus)
    assert np.allclose(total, np.eye(4), atol=1e-9)


# ---- Depolarizing channel ----

def test_depolarizing_channel_fidelity():
    """Depolarizing rate p gives F_avg = 1 - p·d/(d+1) (here approximate)."""
    p = 0.2
    n = 2
    d = 2 ** n
    def depo(rho):
        return (1 - p) * rho + p * np.eye(d, dtype=complex) / d * np.trace(rho)
    J = multi_qubit_process_tomography(depo, n_qubits=n)
    F = average_gate_fidelity_channel(J, np.eye(d, dtype=complex))
    # F_pro = (1-p) + p/d²; F_avg = (d·F_pro + 1)/(d+1) = 1 - p(d-1)/d.
    expected = 1 - p * (d - 1) / d
    assert abs(F - expected) < 1e-9


# ---- Process fidelity properties ----

def test_process_fidelity_self_is_one():
    """F_pro(ε, ε) should equal Tr(J²)/d² which is just |J|²/d² —
    use the identity channel's max-eigenvalue Choi to keep things normalized."""
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=complex)
    J = kraus_to_choi([CNOT])
    F = process_fidelity_choi(J, J)
    # For a unitary channel, Tr(J²) = d² and F_pro = 1.
    assert abs(F - 1.0) < 1e-9
