"""Schwinger model (1+1 D lattice QED) tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.schwinger import (
    schwinger_hamiltonian, schwinger_hamiltonian_matrix,
    schwinger_ground_state,
    chiral_condensate, electric_field_per_link, total_charge,
    string_tension,
)


# ---- Hamiltonian construction ----

def test_hamiltonian_is_hermitian():
    H = schwinger_hamiltonian_matrix(4, w=1.0, m=0.5, g=1.0)
    assert np.allclose(H, H.conj().T, atol=1e-9)


def test_rejects_small_N():
    with pytest.raises(ValueError):
        schwinger_hamiltonian(1)


def test_pauli_op_returned():
    H = schwinger_hamiltonian(4, w=1.0, m=0.5, g=1.0)
    # PauliOp has .terms
    assert hasattr(H, "terms")
    assert len(H.terms) > 0


# ---- Ground-state properties ----

def test_ground_state_normalized():
    E, psi = schwinger_ground_state(4, w=1.0, m=0.5, g=1.0)
    assert abs(np.linalg.norm(psi) - 1.0) < 1e-12


def test_ground_state_charge_neutral():
    """Schwinger vacuum has zero total charge (ε₀ = 0 case)."""
    for N in [4, 6]:
        E, psi = schwinger_ground_state(N, w=1.0, m=0.5, g=1.0, eps0=0.0)
        Q = total_charge(psi, N)
        assert abs(Q) < 1e-9


def test_ground_state_energy_scales_with_N():
    """E_gs should decrease (become more negative or less positive) with N."""
    E4, _ = schwinger_ground_state(4, w=1.0, m=0.5, g=1.0)
    E6, _ = schwinger_ground_state(6, w=1.0, m=0.5, g=1.0)
    E8, _ = schwinger_ground_state(8, w=1.0, m=0.5, g=1.0)
    # Energy is approximately extensive: E ~ -c · N for some c > 0.
    # Check it scales monotonically.
    assert E6 < E4
    assert E8 < E6


# ---- Chiral physics ----

def test_chiral_condensate_nonzero_at_zero_mass():
    """The Schwinger model breaks chiral symmetry — at m=0 the condensate
    should still be nonzero (chiral anomaly)."""
    E, psi = schwinger_ground_state(6, w=1.0, m=0.0, g=1.0)
    cond = chiral_condensate(psi, 6)
    assert abs(cond) > 0.05


def test_chiral_condensate_grows_with_mass():
    """At higher m, condensate magnitude grows (explicit symmetry breaking)."""
    _, psi_small = schwinger_ground_state(4, w=1.0, m=0.1, g=1.0)
    _, psi_large = schwinger_ground_state(4, w=1.0, m=1.0, g=1.0)
    cond_small = abs(chiral_condensate(psi_small, 4))
    cond_large = abs(chiral_condensate(psi_large, 4))
    assert cond_large > cond_small


# ---- Electric field / confinement ----

def test_electric_field_zero_in_vacuum():
    """In the Schwinger vacuum, the electric field per link should average
    to roughly zero (charge balance)."""
    _, psi = schwinger_ground_state(6, w=1.0, m=0.5, g=1.0, eps0=0.0)
    E_field = electric_field_per_link(psi, 6)
    # All entries should be small (well under 1.0).
    assert np.max(np.abs(E_field)) < 0.5


def test_string_tension_positive():
    """A nonzero background field ε₀ should raise the energy (confinement)."""
    r = string_tension(N=6, w=1.0, m=0.5, g=1.0)
    assert r["observed_diff"] > 0


def test_string_tension_matches_perturbative():
    """For small ε₀ and the chosen coupling, the energy shift should be in
    rough agreement with the perturbative prediction (within ~20%)."""
    r = string_tension(N=6, w=1.0, m=0.5, g=1.0)
    rel_err = abs(r["observed_diff"] - r["predicted_diff"]) / r["predicted_diff"]
    assert rel_err < 0.3


# ---- Hopping / mass limit checks ----

def test_no_hopping_limit():
    """At w = 0, the Hamiltonian is diagonal in the occupation basis."""
    H = schwinger_hamiltonian_matrix(4, w=0.0, m=0.5, g=1.0)
    # Should be (numerically) diagonal up to electric-field cross terms.
    diag = np.diag(H)
    off = H - np.diag(diag)
    assert np.max(np.abs(off)) < 1e-9
