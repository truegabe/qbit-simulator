"""Hamiltonian simulation via QSP/QSVT tests."""

import numpy as np
import pytest
from scipy.linalg import expm

from qbit_simulator.algorithms.hamsim_qsvt import (
    truncate_hamsim_polynomial,
    hamsim_via_chebyshev,
    hamsim_error_bound,
    simulate_evolution,
)


# ---- Degree selection ----

def test_truncate_polynomial_grows_with_time():
    d_small = truncate_hamsim_polynomial(t=0.5, eps=1e-8)
    d_large = truncate_hamsim_polynomial(t=5.0, eps=1e-8)
    assert d_large > d_small


def test_truncate_polynomial_grows_with_precision():
    d_loose = truncate_hamsim_polynomial(t=1.0, eps=1e-4)
    d_tight = truncate_hamsim_polynomial(t=1.0, eps=1e-12)
    assert d_tight > d_loose


def test_truncate_zero_time():
    assert truncate_hamsim_polynomial(t=0.0) == 0


# ---- Chebyshev expansion ----

def test_zero_time_is_identity():
    H = np.array([[0.5, 0.3], [0.3, -0.2]], dtype=complex)
    U = hamsim_via_chebyshev(H, t=0.0, max_degree=10)
    assert np.allclose(U, np.eye(2), atol=1e-9)


def test_chebyshev_matches_expm_small_t():
    H = np.array([[0.5, 0.3], [0.3, -0.2]], dtype=complex)
    t = 0.5
    U_true = expm(-1j * H * t)
    U_qsp = hamsim_via_chebyshev(H, t, max_degree=20)
    assert np.allclose(U_true, U_qsp, atol=1e-12)


@pytest.mark.parametrize("t", [0.5, 1.0, 2.0, 4.0])
def test_chebyshev_matches_expm(t):
    H = np.array([[0.5, 0.3], [0.3, -0.2]], dtype=complex)
    U_true = expm(-1j * H * t)
    d = truncate_hamsim_polynomial(t, eps=1e-12)
    U_qsp = hamsim_via_chebyshev(H, t, max_degree=d)
    assert np.allclose(U_true, U_qsp, atol=1e-9)


def test_chebyshev_is_unitary():
    H = np.array([[0.5, 0.3], [0.3, -0.2]], dtype=complex)
    U = hamsim_via_chebyshev(H, t=1.0, max_degree=20)
    assert np.allclose(U @ U.conj().T, np.eye(2), atol=1e-9)


def test_chebyshev_random_hermitian():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    H = 0.5 * (A + A.conj().T)
    # Normalize.
    H /= np.linalg.norm(H, ord=2) * 1.01
    t = 0.7
    U_true = expm(-1j * H * t)
    U_qsp = hamsim_via_chebyshev(H, t, max_degree=25)
    assert np.allclose(U_true, U_qsp, atol=1e-9)


# ---- Error bound ----

def test_error_bound_decreasing_with_degree():
    e_10 = hamsim_error_bound(t=2.0, d=10)
    e_20 = hamsim_error_bound(t=2.0, d=20)
    e_30 = hamsim_error_bound(t=2.0, d=30)
    assert e_30 < e_20 < e_10


def test_error_bound_nonnegative():
    assert hamsim_error_bound(t=1.0, d=10) >= 0


# ---- State evolution ----

def test_evolution_preserves_norm():
    H = np.array([[0.5, 0.3], [0.3, -0.2]], dtype=complex)
    psi0 = np.array([0.6, 0.8], dtype=complex)
    psi_t = simulate_evolution(H, psi0, t=1.5)
    assert abs(np.linalg.norm(psi_t) - 1.0) < 1e-9


def test_evolution_matches_expm():
    H = np.array([[0.5, 0.3], [0.3, -0.2]], dtype=complex)
    psi0 = np.array([1.0, 0.0], dtype=complex)
    t = 1.0
    psi_true = expm(-1j * H * t) @ psi0
    psi_qsp = simulate_evolution(H, psi0, t)
    assert np.allclose(psi_true, psi_qsp, atol=1e-9)


def test_evolution_zero_hamiltonian():
    """exp(-i · 0 · t) |ψ⟩ = |ψ⟩."""
    H = np.zeros((2, 2), dtype=complex)
    psi0 = np.array([0.6, 0.8], dtype=complex)
    psi_t = simulate_evolution(H, psi0, t=10.0)
    assert np.allclose(psi_t, psi0, atol=1e-9)


def test_evolution_rescales_large_hamiltonian():
    """If ||H|| > 1, the code should still work via rescaling."""
    H = np.array([[3.0, 1.0], [1.0, -2.0]], dtype=complex)
    psi0 = np.array([1.0, 0.0], dtype=complex)
    t = 0.5
    psi_true = expm(-1j * H * t) @ psi0
    psi_qsp = simulate_evolution(H, psi0, t)
    assert np.allclose(psi_true, psi_qsp, atol=1e-7)
