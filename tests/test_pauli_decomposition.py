"""Pauli-decomposition utility tests."""

import numpy as np
import pytest

from qbit_simulator.pauli_decomposition import (
    pauli_string_matrix, decompose, reconstruct, decomposition_error,
    pauli_weight, pauli_weight_distribution,
    is_diagonal_pauli_string,
    qubit_wise_commute, commuting_pauli_groups,
    measurement_basis_for_group,
    shot_cost_naive, shot_cost_grouped,
)


# ---- Matrix construction ----

def test_pauli_string_identity():
    M = pauli_string_matrix("II")
    assert np.allclose(M, np.eye(4), atol=1e-12)


def test_pauli_string_x_kron_z():
    XZ = pauli_string_matrix("XZ")
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    assert np.allclose(XZ, np.kron(X, Z), atol=1e-12)


# ---- Decomposition ----

def test_decompose_identity():
    """Identity → just one term, coefficient 1.0."""
    terms = decompose(np.eye(4, dtype=complex))
    assert len(terms) == 1
    coef, s = terms[0]
    assert s == "II"
    assert abs(coef - 1.0) < 1e-12


def test_decompose_pauli_x():
    """X → just X with coefficient 1."""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    terms = decompose(X)
    assert len(terms) == 1
    assert terms[0] == (1 + 0j, "X")


def test_decompose_sum_xy():
    """A = X + Y → two terms."""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    terms = decompose(X + Y)
    assert len(terms) == 2
    found = {s: coef for coef, s in terms}
    assert abs(found["X"] - 1.0) < 1e-12
    assert abs(found["Y"] - 1.0) < 1e-12


def test_decompose_rejects_non_power_of_2():
    with pytest.raises(ValueError):
        decompose(np.eye(3, dtype=complex))


def test_decompose_rejects_non_square():
    with pytest.raises(ValueError):
        decompose(np.zeros((2, 4), dtype=complex))


# ---- Round-trip ----

def test_reconstruct_round_trip_random_2q():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    H = (A + A.conj().T) / 2
    terms = decompose(H, tol=0)   # all 16 terms
    H_rec = reconstruct(terms)
    assert np.allclose(H, H_rec, atol=1e-12)


def test_reconstruction_error_zero_for_random():
    rng = np.random.default_rng(0)
    M = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    assert decomposition_error(M) < 1e-10


def test_reconstruct_empty_raises():
    with pytest.raises(ValueError):
        reconstruct([])


# ---- Hermitian → real coefficients ----

def test_hermitian_has_real_coefficients():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    H = (A + A.conj().T) / 2
    terms = decompose(H, tol=1e-12)
    for coef, _ in terms:
        assert abs(coef.imag) < 1e-10


# ---- Weight diagnostics ----

def test_pauli_weight_count():
    assert pauli_weight("II") == 0
    assert pauli_weight("XYI") == 2
    assert pauli_weight("ZZZZ") == 4


def test_weight_distribution_identity_is_weight_zero():
    dist = pauli_weight_distribution(np.eye(4, dtype=complex))
    assert dist == {0: 1}


def test_weight_distribution_heisenberg_term():
    """Heisenberg term XX + YY + ZZ on 2 qubits: 3 weight-2 strings."""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    H = np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z)
    dist = pauli_weight_distribution(H)
    assert dist == {2: 3}


# ---- Diagonal-string check ----

def test_diagonal_pauli_string():
    assert is_diagonal_pauli_string("IZZI")
    assert is_diagonal_pauli_string("ZZZ")
    assert not is_diagonal_pauli_string("XZ")
    assert not is_diagonal_pauli_string("YI")


# ---- Qubit-wise commute ----

def test_qwc_identity_commutes_with_anything():
    assert qubit_wise_commute("II", "XY")
    assert qubit_wise_commute("ZX", "II")


def test_qwc_same_paulis_commute():
    assert qubit_wise_commute("XY", "XY")
    assert qubit_wise_commute("XZ", "XZ")


def test_qwc_different_paulis_dont_commute():
    assert not qubit_wise_commute("X", "Y")
    assert not qubit_wise_commute("XZ", "YZ")


def test_qwc_one_identity_qubit_ok():
    """If qubit q is I in one string and X in another, they QWC-commute."""
    assert qubit_wise_commute("IZ", "XZ")
    assert qubit_wise_commute("XI", "XY")


# ---- Grouping ----

def test_grouping_single_string_one_group():
    groups = commuting_pauli_groups(["XY"])
    assert groups == [["XY"]]


def test_grouping_qwc_strings_one_group():
    """ZZ and IZ are QWC → one group."""
    groups = commuting_pauli_groups(["ZZ", "IZ", "ZI"])
    assert len(groups) == 1


def test_grouping_non_commuting_creates_separate_groups():
    """XX and YY are not QWC → two groups."""
    groups = commuting_pauli_groups(["XX", "YY"])
    assert len(groups) == 2


# ---- Measurement basis ----

def test_measurement_basis_for_z_group():
    basis = measurement_basis_for_group(["ZZ", "ZI", "IZ"])
    assert basis == "ZZ"


def test_measurement_basis_for_xyz():
    basis = measurement_basis_for_group(["XYZ", "IYZ"])
    assert basis == "XYZ"


# ---- Shot-cost helpers ----

def test_grouped_cost_less_than_naive_for_diagonal_terms():
    """A Hamiltonian with many Z-only terms should be groupable into 1
    measurement group → grouped cost ≪ naive."""
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    # H = Z⊗I + I⊗Z + Z⊗Z (all Z-diagonal, all QWC-commute).
    H = np.kron(Z, I) + np.kron(I, Z) + np.kron(Z, Z)
    terms = decompose(H, tol=1e-9)
    coefs = [c for c, _ in terms]
    strings = [s for _, s in terms]
    naive = shot_cost_naive(coefs, shots_per_term=100)
    grouped = shot_cost_grouped(coefs, strings, shots_per_group=100)
    assert grouped < naive
    assert grouped == 100   # all in one group
