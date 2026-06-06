"""VQLS tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.vqls import vqls


def test_vqls_diagonal_2x2():
    """VQLS on a diagonal 2x2 system."""
    A = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.complex128)
    b = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2)
    result = vqls(A, b, n_layers=2, max_iter=200, seed=0)
    assert result["fidelity"] > 0.95


def test_vqls_nondiagonal_2x2():
    A = np.array([[2.0, 0.5], [0.5, 1.5]], dtype=np.complex128)
    b = np.array([1.0, 0.0], dtype=np.complex128)
    result = vqls(A, b, n_layers=3, max_iter=300, seed=0)
    assert result["fidelity"] > 0.9


def test_vqls_cost_decreases():
    """The optimizer should decrease the cost."""
    A = np.array([[2.0, 0.5], [0.5, 1.5]], dtype=np.complex128)
    b = np.array([1.0, 0.0], dtype=np.complex128)
    result = vqls(A, b, n_layers=3, max_iter=200, seed=0)
    # A good cost should be near zero.
    assert result["cost_final"] < 0.1


def test_vqls_rejects_wrong_shape_b():
    A = np.eye(2, dtype=np.complex128)
    b = np.array([1.0, 0.0, 0.0], dtype=np.complex128)
    with pytest.raises(ValueError):
        vqls(A, b)


def test_vqls_4x4_system():
    """VQLS on a 4x4 Hermitian system (2 qubits)."""
    rng = np.random.default_rng(0)
    M = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    A = (M + M.conj().T) / 2 + 4 * np.eye(4)   # diagonally dominant
    b = np.array([1, 0.5, 0.3, 0.2], dtype=np.complex128)
    b /= np.linalg.norm(b)
    result = vqls(A, b, n_layers=4, max_iter=400, seed=1)
    assert result["fidelity"] > 0.85
