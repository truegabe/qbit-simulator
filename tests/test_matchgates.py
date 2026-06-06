"""Matchgate / free-fermion circuit tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.matchgates import (
    matchgate, random_matchgate, is_matchgate,
    initial_correlation_matrix, matchgate_to_so2n_block,
    simulate_free_fermion_circuit, occupation_from_gamma,
    majorana_correlation,
)


# ---- Matchgate construction ----

def test_matchgate_identity():
    I = np.eye(2, dtype=complex)
    M = matchgate(I, I)
    assert np.allclose(M, np.eye(4), atol=1e-12)


def test_matchgate_structure():
    """The off-corner blocks should be zero."""
    I = np.eye(2, dtype=complex)
    M = matchgate(I, I)
    # Off positions should be 0.
    for i, j in [(0, 1), (0, 2), (1, 0), (1, 3),
                  (2, 0), (2, 3), (3, 1), (3, 2)]:
        assert abs(M[i, j]) < 1e-12


def test_matchgate_rejects_wrong_shape():
    with pytest.raises(ValueError):
        matchgate(np.zeros(2), np.eye(2))


def test_matchgate_rejects_mismatched_det():
    A = np.array([[1, 0], [0, 1]], dtype=complex)
    B = np.array([[2, 0], [0, 1]], dtype=complex)   # det = 2 ≠ 1
    with pytest.raises(ValueError):
        matchgate(A, B)


# ---- is_matchgate ----

def test_is_matchgate_identity():
    assert is_matchgate(np.eye(4, dtype=complex))


def test_is_matchgate_rejects_random_unitary():
    """A generic 4×4 unitary is NOT a matchgate."""
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    assert not is_matchgate(Q)


def test_is_matchgate_wrong_shape():
    assert not is_matchgate(np.eye(2, dtype=complex))


# ---- Random matchgate ----

def test_random_matchgate_is_matchgate():
    rng = np.random.default_rng(0)
    for _ in range(5):
        M = random_matchgate(rng)
        assert is_matchgate(M)


def test_random_matchgate_unitary():
    rng = np.random.default_rng(0)
    for _ in range(5):
        M = random_matchgate(rng)
        assert np.allclose(M @ M.conj().T, np.eye(4), atol=1e-9)


# ---- Initial correlation matrix ----

def test_initial_correlation_vacuum():
    """Vacuum: ⟨γ_{2k} γ_{2k+1}⟩ = +1 for all k."""
    Gamma = initial_correlation_matrix(3)
    for k in range(3):
        assert abs(Gamma[2 * k, 2 * k + 1] - 1.0) < 1e-12


def test_initial_correlation_antisymmetric():
    Gamma = initial_correlation_matrix(3)
    assert np.allclose(Gamma, -Gamma.T, atol=1e-12)


def test_initial_correlation_with_occupied():
    """An occupied mode k has Γ_{2k, 2k+1} = -1."""
    Gamma = initial_correlation_matrix(3, occupied=[1])
    assert Gamma[2, 3] == -1.0
    assert Gamma[0, 1] == 1.0      # mode 0 still vacuum


# ---- SO(2n) block ----

def test_so2n_block_orthogonal_for_random_matchgate():
    rng = np.random.default_rng(0)
    M = random_matchgate(rng)
    R = matchgate_to_so2n_block(M)
    assert np.allclose(R @ R.T, np.eye(4), atol=1e-9)


def test_so2n_block_identity_for_identity_gate():
    R = matchgate_to_so2n_block(np.eye(4, dtype=complex))
    assert np.allclose(R, np.eye(4), atol=1e-12)


# ---- Free-fermion simulation ----

def test_simulate_empty_circuit_preserves_state():
    Gamma_init = initial_correlation_matrix(3, occupied=[0, 2])
    Gamma_final = simulate_free_fermion_circuit(3, gate_list=[], initial_occupied=[0, 2])
    assert np.allclose(Gamma_init, Gamma_final, atol=1e-12)


def test_simulate_hopping_transfers_occupation():
    """A full-swap matchgate exchanges modes."""
    swap = matchgate(np.array([[0, 1], [1, 0]], dtype=complex),
                       np.array([[0, 1], [1, 0]], dtype=complex))
    Gamma = simulate_free_fermion_circuit(
        4, gate_list=[(swap, 0)], initial_occupied=[0]
    )
    assert abs(occupation_from_gamma(Gamma, 0) - 0.0) < 1e-9
    assert abs(occupation_from_gamma(Gamma, 1) - 1.0) < 1e-9


def test_simulate_rejects_non_4x4():
    with pytest.raises(ValueError):
        simulate_free_fermion_circuit(3, gate_list=[(np.eye(2), 0)])


def test_simulate_rejects_out_of_range_qubit():
    with pytest.raises(ValueError):
        simulate_free_fermion_circuit(3, gate_list=[(np.eye(4), 5)])


def test_gamma_remains_antisymmetric_after_evolution():
    """Free-fermion evolution preserves antisymmetry of Γ (it's
    similarity-transformed by an orthogonal matrix)."""
    rng = np.random.default_rng(0)
    n = 6
    gates = [(random_matchgate(rng), rng.integers(0, n - 1)) for _ in range(10)]
    Gamma = simulate_free_fermion_circuit(
        n, gates, initial_occupied=[0, 2, 4]
    )
    assert np.allclose(Gamma, -Gamma.T, atol=1e-9)


def test_gamma_spectrum_preserved_under_evolution():
    """Γ → R Γ Rᵀ for orthogonal R: eigenvalues are preserved."""
    rng = np.random.default_rng(0)
    n = 4
    Gamma_init = initial_correlation_matrix(n, occupied=[0, 2])
    gates = [(random_matchgate(rng), rng.integers(0, n - 1)) for _ in range(5)]
    Gamma_final = simulate_free_fermion_circuit(
        n, gates, initial_occupied=[0, 2],
    )
    eigs_init = sorted(np.linalg.eigvals(Gamma_init).imag)
    eigs_final = sorted(np.linalg.eigvals(Gamma_final).imag)
    assert np.allclose(eigs_init, eigs_final, atol=1e-8)


def test_majorana_correlation_indexing():
    Gamma = initial_correlation_matrix(2, occupied=[0])
    val = majorana_correlation(Gamma, 0, 1)
    # Mode 0 occupied → Γ_{0,1} = -1.
    assert val == -1.0
