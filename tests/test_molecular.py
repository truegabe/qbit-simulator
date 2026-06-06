"""Tests for the generic molecular-Hamiltonian builder."""

import numpy as np
import pytest

from qbit_simulator.algorithms.molecular import (
    molecular_hamiltonian, molecular_hamiltonian_pauli,
    project_to_n_electron_sector,
    h2_sto3g_mo_integrals, h2_sto3g_full_hamiltonian, h2_full_fci_energy,
    lih_sto3g_integrals_stub,
)
from qbit_simulator.algorithms.h2_sto3g import h2_sto3g_energy


# ---- Generic builder ----

def test_molecular_hamiltonian_returns_fermion_op():
    h = np.array([[-1.0]])
    eri = np.zeros((1, 1, 1, 1))
    H = molecular_hamiltonian(h, eri, V_nn=0.0)
    assert hasattr(H, "to_pauli_op")


def test_zero_integrals_gives_nuclear_only():
    h = np.zeros((2, 2))
    eri = np.zeros((2, 2, 2, 2))
    H_pauli = molecular_hamiltonian_pauli(h, eri, V_nn=1.5)
    # Should be just 1.5 · I.
    assert len(H_pauli.terms) == 1
    coef, s = H_pauli.terms[0]
    assert abs(coef - 1.5) < 1e-12
    assert s == "IIII"


# ---- Sector projection ----

def test_project_to_n_electron_sector_dimensions():
    """For n_qubits=4, the 2-electron sector has C(4,2) = 6 basis states."""
    H = np.eye(16)
    H_2e, indices = project_to_n_electron_sector(H, 4, 2)
    assert H_2e.shape == (6, 6)
    assert len(indices) == 6


def test_project_to_zero_electron_sector():
    H = np.eye(16)
    H_0, indices = project_to_n_electron_sector(H, 4, 0)
    assert H_0.shape == (1, 1)
    assert indices == [0]


# ---- H₂ STO-3G full vs reduced ----

@pytest.mark.parametrize("R", [0.5, 0.74, 1.0, 1.5, 2.0])
def test_h2_full_matches_reduced(R):
    """The 4-qubit fermionic H₂ STO-3G should give the same FCI energy
    as the 2-qubit reduced version."""
    E_full = h2_full_fci_energy(R)
    E_reduced = h2_sto3g_energy(R)
    assert abs(E_full - E_reduced) < 1e-10


def test_h2_integrals_shape():
    ints = h2_sto3g_mo_integrals(0.74)
    assert ints["h_mo"].shape == (2, 2)
    assert ints["eri_mo"].shape == (2, 2, 2, 2)
    assert isinstance(ints["V_nn"], float)


def test_h2_full_hamiltonian_has_pauli_terms():
    H = h2_sto3g_full_hamiltonian(0.74)
    assert len(H.terms) > 5    # nontrivial structure
    # All terms are 4-character Pauli strings.
    for coef, s in H.terms:
        assert len(s) == 4


def test_h2_full_hamiltonian_hermitian():
    from qbit_simulator.algorithms.ucc import _pauli_op_to_matrix
    H_pauli = h2_sto3g_full_hamiltonian(0.74)
    H_mat = _pauli_op_to_matrix(H_pauli, n=4)
    assert np.allclose(H_mat, H_mat.conj().T, atol=1e-9)


# ---- LiH stub ----

def test_lih_stub_has_right_shapes():
    ints = lih_sto3g_integrals_stub()
    assert ints["h_mo"].shape == (3, 3)
    assert ints["eri_mo"].shape == (3, 3, 3, 3)
    assert "stub" in ints and ints["stub"]


def test_lih_stub_builds_6_qubit_hamiltonian():
    ints = lih_sto3g_integrals_stub()
    H = molecular_hamiltonian_pauli(
        ints["h_mo"], ints["eri_mo"], ints["V_nn"],
    )
    # 3 spatial × 2 spin = 6 qubits, so each Pauli string has length 6.
    for coef, s in H.terms:
        assert len(s) == 6


def test_lih_stub_h_is_symmetric():
    ints = lih_sto3g_integrals_stub()
    h = ints["h_mo"]
    assert np.allclose(h, h.T, atol=1e-12)
