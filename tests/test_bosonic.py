"""Bosonic codes tests (cat and GKP)."""

import numpy as np
import pytest

from qbit_simulator.bosonic import (
    annihilation_operator, creation_operator, number_operator,
    position_operator, momentum_operator,
    coherent_state, displacement_operator,
    cat_state, cat_parity, photon_loss,
    gkp_state, gkp_x_expectation,
)


# ---- Fock-space operators ----

def test_annihilation_acts_correctly():
    """a|n⟩ = √n |n-1⟩."""
    a = annihilation_operator(5)
    state_3 = np.zeros(5)
    state_3[3] = 1.0
    out = a @ state_3
    assert abs(out[2] - np.sqrt(3)) < 1e-12
    assert abs(out[0]) < 1e-12


def test_commutator_a_adag():
    """[a, a†] = I (within truncation)."""
    D = 10
    a = annihilation_operator(D)
    adag = creation_operator(D)
    comm = a @ adag - adag @ a
    # All diagonal entries should be 1 except the last (truncation).
    for k in range(D - 1):
        assert abs(comm[k, k] - 1.0) < 1e-12


def test_number_operator():
    N = number_operator(5)
    state_3 = np.zeros(5, dtype=complex); state_3[3] = 1.0
    assert abs((N @ state_3)[3] - 3.0) < 1e-12


def test_position_eigenvalue_zero():
    """⟨x⟩ on |n=0⟩ is zero by symmetry."""
    x = position_operator(10)
    vac = np.zeros(10, dtype=complex); vac[0] = 1.0
    val = float(np.real(vac.conj() @ x @ vac))
    assert abs(val) < 1e-12


# ---- Coherent states ----

def test_coherent_state_normalized():
    psi = coherent_state(2.0, 30)
    assert abs(np.linalg.norm(psi) - 1.0) < 1e-10


def test_coherent_state_mean_photon_number():
    """⟨n⟩ = |α|²."""
    alpha = 1.5
    psi = coherent_state(alpha, 40)
    N = number_operator(40)
    mean = float(np.real(psi.conj() @ N @ psi))
    assert abs(mean - alpha ** 2) < 1e-6


def test_coherent_eigenstate_of_a():
    """a|α⟩ = α|α⟩."""
    alpha = 1.2
    psi = coherent_state(alpha, 40)
    a = annihilation_operator(40)
    out = a @ psi
    # out should equal alpha · psi (within truncation).
    expected = alpha * psi
    # Compare first D-3 entries (truncation effect at high n).
    assert np.allclose(out[:-3], expected[:-3], atol=1e-9)


# ---- Cat states ----

def test_cat_states_normalized():
    psi0 = cat_state(2.0, 0, 30)
    psi1 = cat_state(2.0, 1, 30)
    assert abs(np.linalg.norm(psi0) - 1.0) < 1e-12
    assert abs(np.linalg.norm(psi1) - 1.0) < 1e-12


def test_cat_parity_eigenvalues():
    """Even cat has parity +1, odd cat has parity -1."""
    psi0 = cat_state(2.0, 0, 30)
    psi1 = cat_state(2.0, 1, 30)
    assert abs(cat_parity(psi0) - 1.0) < 1e-6
    assert abs(cat_parity(psi1) + 1.0) < 1e-6


def test_cat_states_orthogonal():
    psi0 = cat_state(2.0, 0, 30)
    psi1 = cat_state(2.0, 1, 30)
    overlap = abs(np.vdot(psi0, psi1))
    assert overlap < 1e-9


# ---- Photon loss channel ----

def test_photon_loss_preserves_trace():
    psi = cat_state(2.0, 0, 30)
    rho = np.outer(psi, psi.conj())
    rho_after = photon_loss(rho, gamma=0.3)
    assert abs(np.trace(rho_after).real - 1.0) < 1e-8


def test_photon_loss_zero_is_identity():
    psi = cat_state(2.0, 0, 30)
    rho = np.outer(psi, psi.conj())
    rho_after = photon_loss(rho, gamma=0.0)
    assert np.allclose(rho, rho_after, atol=1e-12)


def test_photon_loss_full_is_vacuum():
    """γ = 1: any state decays to the vacuum."""
    psi = cat_state(2.0, 0, 30)
    rho = np.outer(psi, psi.conj())
    rho_after = photon_loss(rho, gamma=1.0)
    # rho_after[0, 0] should be ~1 (vacuum).
    assert abs(rho_after[0, 0].real - 1.0) < 1e-9


def test_photon_loss_decays_parity():
    """Each photon loss flips parity → parity expectation decays."""
    psi = cat_state(2.0, 0, 30)
    rho = np.outer(psi, psi.conj())
    parity_initial = cat_parity(rho)
    parity_after = cat_parity(photon_loss(rho, gamma=0.3))
    # Parity should still be positive but smaller than 1.
    assert 0 < parity_after < parity_initial


def test_photon_loss_invalid_gamma():
    psi = cat_state(2.0, 0, 30)
    rho = np.outer(psi, psi.conj())
    with pytest.raises(ValueError):
        photon_loss(rho, gamma=1.5)


# ---- Displacement ----

def test_displacement_zero_is_identity():
    D_op = displacement_operator(0.0, 10)
    assert np.allclose(D_op, np.eye(10), atol=1e-9)


def test_displacement_creates_coherent_from_vacuum():
    """D(α)|0⟩ = |α⟩."""
    alpha = 1.0
    D = 30
    D_op = displacement_operator(alpha, D)
    vac = np.zeros(D, dtype=complex); vac[0] = 1.0
    psi = D_op @ vac
    psi_coh = coherent_state(alpha, D)
    assert np.allclose(psi, psi_coh, atol=1e-9)


# ---- GKP states ----

def test_gkp_states_normalized():
    psi0 = gkp_state(0, sigma=3.0, n_terms=8, D=80)
    psi1 = gkp_state(1, sigma=3.0, n_terms=8, D=80)
    assert abs(np.linalg.norm(psi0) - 1.0) < 1e-10
    assert abs(np.linalg.norm(psi1) - 1.0) < 1e-10


def test_gkp_x_expectation_zero():
    """GKP states are symmetric → ⟨x⟩ = 0."""
    psi0 = gkp_state(0, sigma=3.0, n_terms=8, D=80)
    psi1 = gkp_state(1, sigma=3.0, n_terms=8, D=80)
    assert abs(gkp_x_expectation(psi0)) < 1e-8
    assert abs(gkp_x_expectation(psi1)) < 1e-8


def test_gkp_states_distinguishable():
    """GKP |0⟩ and |1⟩ should give opposite-sign Z stabilizer."""
    D = 80
    psi0 = gkp_state(0, sigma=3.0, n_terms=8, D=D)
    psi1 = gkp_state(1, sigma=3.0, n_terms=8, D=D)
    x = position_operator(D)
    eigs, V = np.linalg.eigh(x.astype(complex))
    Z_op = V @ np.diag(np.cos(np.sqrt(np.pi) * eigs)) @ V.conj().T
    Z0 = float(np.real(psi0.conj() @ Z_op @ psi0))
    Z1 = float(np.real(psi1.conj() @ Z_op @ psi1))
    # Z0 should be substantially positive, Z1 substantially negative.
    assert Z0 > 0.2
    assert Z1 < -0.2


def test_gkp_invalid_parity():
    with pytest.raises(ValueError):
        gkp_state(2, sigma=3.0)
