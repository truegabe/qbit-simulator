"""Block encoding + LCU tests."""

import numpy as np
import pytest
from scipy.linalg import expm

from qbit_simulator.algorithms.block_encoding import (
    block_encode_lcu, extract_block_encoded, apply_block_encoded,
    truncated_taylor_simulation,
)


_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_I = np.eye(2, dtype=complex)


# ---- block_encode_lcu ----

def test_lcu_block_is_unitary():
    U, _ = block_encode_lcu([0.5, 0.5], [_X, _Z])
    assert np.allclose(U @ U.conj().T, np.eye(U.shape[0]), atol=1e-9)


def test_lcu_block_extracts_correct_matrix():
    """For A = 0.5 X + 0.5 Z, the extracted block = A / 1.0."""
    A_target = 0.5 * _X + 0.5 * _Z
    U, alpha = block_encode_lcu([0.5, 0.5], [_X, _Z])
    A_block = extract_block_encoded(U, n_ancilla=1)
    assert np.allclose(A_block, A_target / alpha, atol=1e-9)


def test_lcu_single_term():
    """A single-term LCU should give back the input unitary."""
    U, alpha = block_encode_lcu([2.0], [_X])
    A_block = extract_block_encoded(U, n_ancilla=0)
    # With one term, no ancilla needed; U is just X.
    assert np.allclose(A_block, _X, atol=1e-9)
    assert abs(alpha - 2.0) < 1e-9


def test_lcu_handles_negative_coeffs():
    """Negative α should be folded into the unitary."""
    U, alpha = block_encode_lcu([0.5, -0.5], [_X, _Z])
    A_block = extract_block_encoded(U, n_ancilla=1)
    # |-0.5| absorbed → A = 0.5·X + 0.5·(-Z) = 0.5(X - Z).
    expected = (0.5 * _X - 0.5 * _Z) / alpha
    assert np.allclose(A_block, expected, atol=1e-9)


def test_lcu_pads_to_power_of_two():
    """3 unitaries → ancilla rounded up to 2 qubits."""
    U, alpha = block_encode_lcu([0.4, 0.3, 0.3], [_X, _Y, _Z])
    # 3 → next power of 2 = 4, so 2 ancilla qubits.
    assert U.shape == (8, 8)


def test_lcu_alpha_is_l1_norm():
    coeffs = [0.3, 0.5, 0.2]
    _, alpha = block_encode_lcu(coeffs, [_X, _Y, _Z])
    assert abs(alpha - 1.0) < 1e-9


def test_lcu_rejects_empty():
    with pytest.raises(ValueError):
        block_encode_lcu([], [])


# ---- apply_block_encoded ----

def test_apply_block_encoded_to_state():
    A_target = 0.5 * _X + 0.5 * _Z
    U, alpha = block_encode_lcu([0.5, 0.5], [_X, _Z])
    psi = np.array([1, 0], dtype=complex)
    a_psi, p = apply_block_encoded(U, n_ancilla=1, psi=psi)
    # Expected: A / alpha applied to psi, then renormalize.
    expected = (A_target / alpha) @ psi
    expected_norm = expected / np.linalg.norm(expected)
    assert np.allclose(a_psi, expected_norm, atol=1e-9)


def test_apply_block_encoded_p_success_in_range():
    U, _ = block_encode_lcu([0.5, 0.5], [_X, _Z])
    psi = np.array([1, 0], dtype=complex)
    _, p = apply_block_encoded(U, n_ancilla=1, psi=psi)
    assert 0.0 <= p <= 1.0


def test_apply_block_encoded_rejects_wrong_shape():
    U, _ = block_encode_lcu([0.5, 0.5], [_X, _Z])
    with pytest.raises(ValueError):
        apply_block_encoded(U, n_ancilla=1, psi=np.zeros(3, dtype=complex))


# ---- Truncated Taylor ----

def test_truncated_taylor_zero_time_is_identity():
    U = truncated_taylor_simulation([(1.0, "X")], t=0.0, k_max=5)
    assert np.allclose(U, np.eye(2), atol=1e-12)


def test_truncated_taylor_matches_expm_small_t():
    H_paulis = [(0.5, "X")]
    H_mat = 0.5 * _X
    t = 0.3
    U_true = expm(-1j * H_mat * t)
    U_tay = truncated_taylor_simulation(H_paulis, t, k_max=10)
    assert np.allclose(U_true, U_tay, atol=1e-10)


def test_truncated_taylor_multi_pauli():
    """H = 0.3 X + 0.4 Z."""
    H_paulis = [(0.3, "X"), (0.4, "Z")]
    H_mat = 0.3 * _X + 0.4 * _Z
    t = 0.5
    U_true = expm(-1j * H_mat * t)
    U_tay = truncated_taylor_simulation(H_paulis, t, k_max=12)
    assert np.allclose(U_true, U_tay, atol=1e-10)


def test_truncated_taylor_rejects_empty():
    with pytest.raises(ValueError):
        truncated_taylor_simulation([], t=1.0)


def test_truncated_taylor_higher_k_more_accurate():
    H_paulis = [(0.5, "X")]
    H_mat = 0.5 * _X
    t = 0.5
    U_true = expm(-1j * H_mat * t)
    err_3 = np.linalg.norm(truncated_taylor_simulation(H_paulis, t, 3) - U_true)
    err_8 = np.linalg.norm(truncated_taylor_simulation(H_paulis, t, 8) - U_true)
    assert err_8 < err_3
