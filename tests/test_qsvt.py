"""Quantum Singular Value Transformation tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.qsvt import (
    block_encode_hermitian, qsvt_unitary, qsvt_polynomial,
    chebyshev_phases, chebyshev_t_of_matrix,
    chebyshev_expansion_cos, evaluate_chebyshev_polynomial,
)


# ---- block encoding ----

def test_block_encode_is_unitary():
    """The rotation-style block encoding U must be unitary."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 4))
    A = (A + A.T) / 2
    A = A / (1.5 * np.max(np.abs(np.linalg.eigvalsh(A))))   # ||A|| < 1
    U = block_encode_hermitian(A)
    assert np.allclose(U @ U.conj().T, np.eye(8), atol=1e-10)


def test_block_encode_recovers_A():
    """Top-left block of U equals A."""
    A = np.array([[0.3, 0.1], [0.1, -0.2]], dtype=np.complex128)
    U = block_encode_hermitian(A)
    assert np.allclose(U[:2, :2], A, atol=1e-12)


def test_block_encode_rejects_norm_too_large():
    """A with ||A|| > 1 should be rejected."""
    A = np.array([[2.0, 0.0], [0.0, 0.5]], dtype=np.complex128)
    with pytest.raises(ValueError):
        block_encode_hermitian(A)


def test_block_encode_rejects_non_hermitian():
    A = np.array([[0.5, 0.1], [0.0, 0.3]], dtype=np.complex128)
    with pytest.raises(ValueError):
        block_encode_hermitian(A)


# ---- core identity: U_Φ with zero phases gives U^d, block = T_d(A) ----

def test_qsvt_zero_phases_d1_gives_A():
    """phases=[0, 0]: U_Φ = U, block = A = T_1(A)."""
    A = np.array([[0.3, 0.1], [0.1, -0.2]], dtype=np.complex128)
    U = block_encode_hermitian(A)
    block = qsvt_polynomial(U, [0.0, 0.0])
    assert np.allclose(block, A, atol=1e-12)


def test_qsvt_zero_phases_d2_gives_T2():
    """phases=[0]*3: block = T_2(A) = 2A² − I."""
    A = np.array([[0.3, 0.1], [0.1, -0.2]], dtype=np.complex128)
    U = block_encode_hermitian(A)
    block = qsvt_polynomial(U, [0.0, 0.0, 0.0])
    expected = 2 * A @ A - np.eye(2, dtype=np.complex128)
    assert np.allclose(block, expected, atol=1e-10)


@pytest.mark.parametrize("d", [0, 1, 2, 3, 4, 5, 8, 12])
def test_qsvt_zero_phases_chebyshev_recurrence(d):
    """For a random Hermitian A with ||A|| < 1, the QSVT block with
    zero phases equals T_d(A) computed via the Chebyshev recurrence."""
    rng = np.random.default_rng(d)
    M = rng.normal(size=(3, 3))
    A = (M + M.T) / 2
    A = A / (2 * np.max(np.abs(np.linalg.eigvalsh(A))))     # ||A|| = 1/2
    U = block_encode_hermitian(A)
    block = qsvt_polynomial(U, chebyshev_phases(d))
    expected = chebyshev_t_of_matrix(d, A)
    assert np.allclose(block, expected, atol=1e-9)


# ---- random phases produce a unitary block ----

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_qsvt_unitary_for_random_phases(seed):
    """For random phases, U_Φ is unitary."""
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(2, 2))
    A = (M + M.T) / 2
    A = A / (2 * np.max(np.abs(np.linalg.eigvalsh(A))))
    U = block_encode_hermitian(A)
    phases = list(rng.uniform(0, 2 * np.pi, size=8))
    U_phi = qsvt_unitary(U, phases)
    assert np.allclose(U_phi @ U_phi.conj().T, np.eye(U_phi.shape[0]),
                       atol=1e-9)


def test_qsvt_block_norm_bounded():
    """|P(A)| ≤ I on the eigenspectrum, so all singular values of the
    block are ≤ 1."""
    rng = np.random.default_rng(0)
    A = (lambda M: (M + M.T) / 2)(rng.normal(size=(3, 3)))
    A /= 2 * np.max(np.abs(np.linalg.eigvalsh(A)))
    U = block_encode_hermitian(A)
    for _ in range(5):
        phases = list(rng.uniform(0, 2 * np.pi, size=6))
        block = qsvt_polynomial(U, phases)
        sigmas = np.linalg.svd(block, compute_uv=False)
        assert all(s <= 1.0 + 1e-9 for s in sigmas)


# ---- chebyshev_t_of_matrix sanity ----

def test_chebyshev_t_of_matrix_recurrence():
    """T_3(x) = 4x³ - 3x. Check on a scalar A = [[0.5]]."""
    A = np.array([[0.5]], dtype=np.complex128)
    T3 = chebyshev_t_of_matrix(3, A)
    expected = 4 * 0.5 ** 3 - 3 * 0.5
    assert abs(T3[0, 0] - expected) < 1e-12


# ---- chebyshev expansion of cos ----

def test_chebyshev_expansion_cos_evaluates_correctly():
    """For small A, Σ c_n T_n(A) ≈ cos(t·A) up to truncation."""
    rng = np.random.default_rng(0)
    A = (lambda M: (M + M.T) / 2)(rng.normal(size=(2, 2)))
    A /= 4 * np.max(np.abs(np.linalg.eigvalsh(A)))   # ||A|| = 0.25 for fast convergence
    t = 1.5
    coeffs = chebyshev_expansion_cos(t, max_degree=30)
    approx = evaluate_chebyshev_polynomial(coeffs, A)
    # Reference: cos(t · A) computed via eigendecomposition.
    from scipy.linalg import cosm
    expected = cosm(t * A)
    err = np.linalg.norm(approx - expected)
    assert err < 1e-6
