"""Higher-order Trotter-Suzuki tests."""

import numpy as np
import pytest
from scipy.linalg import expm

from qbit_simulator.algorithms.trotter_suzuki import (
    trotter_step_order1, trotter_step_order2, trotter_step_order4,
    trotter_step_order_2k,
    evolve_order, trotter_error, error_scaling_table,
)


_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


# ---- Single-step correctness ----

def test_order1_commuting_is_exact():
    """For commuting H's, Trotter is exact at any order."""
    H_list = [_Z, _Z]  # both Z (commute trivially)
    t = 1.0
    err = trotter_error(H_list, t, n_steps=1, order=1)
    assert err < 1e-10


def test_order1_single_term_exact():
    """Single term: U = exp(-iHt) exactly."""
    H_list = [_X]
    t = 0.7
    err = trotter_error(H_list, t, n_steps=1, order=1)
    assert err < 1e-10


def test_order2_is_unitary():
    H_list = [_X, _Z]
    U = trotter_step_order2(H_list, 0.5)
    assert np.allclose(U @ U.conj().T, np.eye(2), atol=1e-10)


def test_order4_is_unitary():
    H_list = [_X, _Z]
    U = trotter_step_order4(H_list, 0.5)
    assert np.allclose(U @ U.conj().T, np.eye(2), atol=1e-10)


def test_order1_more_steps_lower_error():
    H_list = [_X, _Z]
    e1 = trotter_error(H_list, t=1.0, n_steps=1, order=1)
    e10 = trotter_error(H_list, t=1.0, n_steps=10, order=1)
    assert e10 < e1


# ---- Order scaling ----

def test_order1_scales_as_1_over_N():
    """Doubling N halves the error for order-1 Trotter."""
    H_list = [_X, _Z]
    e_2 = trotter_error(H_list, t=1.0, n_steps=2, order=1)
    e_4 = trotter_error(H_list, t=1.0, n_steps=4, order=1)
    e_8 = trotter_error(H_list, t=1.0, n_steps=8, order=1)
    # Ratios should be approximately 2.
    assert abs(e_2 / e_4 - 2.0) < 0.3
    assert abs(e_4 / e_8 - 2.0) < 0.3


def test_order2_scales_as_1_over_N_squared():
    """Doubling N reduces error by 4 for order-2 Trotter."""
    H_list = [_X, _Z]
    e_2 = trotter_error(H_list, t=1.0, n_steps=2, order=2)
    e_4 = trotter_error(H_list, t=1.0, n_steps=4, order=2)
    e_8 = trotter_error(H_list, t=1.0, n_steps=8, order=2)
    assert abs(e_2 / e_4 - 4.0) < 0.5
    assert abs(e_4 / e_8 - 4.0) < 0.5


def test_order4_scales_as_1_over_N_to_4():
    """Doubling N reduces error by 16 for order-4 Trotter."""
    H_list = [_X, _Z]
    e_2 = trotter_error(H_list, t=1.0, n_steps=2, order=4)
    e_4 = trotter_error(H_list, t=1.0, n_steps=4, order=4)
    # Should be roughly 16, allow factor-of-2 slack.
    assert e_2 / e_4 > 10


def test_order4_better_than_order2():
    """Order-4 should give lower error than order-2 at same step count
    (for non-trivial systems and modest dt)."""
    H_list = [_X, _Z]
    e_2 = trotter_error(H_list, t=1.0, n_steps=4, order=2)
    e_4 = trotter_error(H_list, t=1.0, n_steps=4, order=4)
    assert e_4 < e_2


# ---- General order_2k ----

def test_order_2k_matches_order_2_for_k_eq_1():
    H_list = [_X, _Z]
    U_2 = trotter_step_order2(H_list, 0.3)
    U_2k1 = trotter_step_order_2k(H_list, 0.3, k=1)
    assert np.allclose(U_2, U_2k1, atol=1e-12)


def test_order_2k_matches_order_4_for_k_eq_2():
    H_list = [_X, _Z]
    U_4 = trotter_step_order4(H_list, 0.3)
    U_2k2 = trotter_step_order_2k(H_list, 0.3, k=2)
    assert np.allclose(U_4, U_2k2, atol=1e-12)


def test_order_2k_higher_k_more_accurate():
    """Higher k → lower Trotter error."""
    H_list = [_X, _Z]
    e_k1 = trotter_error(H_list, t=1.0, n_steps=2, order=2)
    e_k2 = trotter_error(H_list, t=1.0, n_steps=2, order=4)
    e_k3 = trotter_error(H_list, t=1.0, n_steps=2, order=6)
    assert e_k1 > e_k2 > e_k3


# ---- evolve_order ----

def test_evolve_order_rejects_odd_order_above_1():
    H_list = [_X, _Z]
    with pytest.raises(ValueError):
        evolve_order(H_list, t=1.0, n_steps=1, order=3)


def test_evolve_order_3q_hamiltonian():
    """Test on a 3-qubit Heisenberg term."""
    XX = np.kron(_X, _X)
    ZZ = np.kron(_Z, _Z)
    H_list = [XX, ZZ]
    U_true = expm(-1j * (XX + ZZ) * 0.5)
    U_t4 = evolve_order(H_list, t=0.5, n_steps=4, order=4)
    assert np.linalg.norm(U_true - U_t4, ord=2) < 1e-5


# ---- error_scaling_table ----

def test_error_scaling_table_format():
    H_list = [_X, _Z]
    table = error_scaling_table(H_list, t=1.0, order=2,
                                  n_steps_list=[2, 4, 8])
    assert len(table) == 3
    assert all(isinstance(N, int) for N, _ in table)
    assert all(err > 0 for _, err in table)
