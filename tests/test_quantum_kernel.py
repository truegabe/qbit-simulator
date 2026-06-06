"""Quantum kernel tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.quantum_kernel import (
    zz_feature_map, quantum_kernel_value, quantum_kernel_matrix,
)


def test_feature_map_returns_unit_state():
    x = np.array([0.5, 0.7])
    qc = zz_feature_map(x)
    assert abs(np.linalg.norm(qc.state) - 1.0) < 1e-10


def test_kernel_self_value_is_one():
    """K(x, x) = 1 by definition (|⟨φ(x)|φ(x)⟩|² = 1)."""
    x = np.array([0.3, 0.5, 0.7])
    assert quantum_kernel_value(x, x) == pytest.approx(1.0, abs=1e-9)


def test_kernel_is_symmetric():
    x = np.array([0.2, 0.4])
    y = np.array([0.7, 0.1])
    assert quantum_kernel_value(x, y) == pytest.approx(
        quantum_kernel_value(y, x), abs=1e-9,
    )


def test_kernel_matrix_diagonal_is_one():
    X = np.random.default_rng(0).uniform(0, np.pi, size=(5, 3))
    K = quantum_kernel_matrix(X)
    for i in range(5):
        assert K[i, i] == pytest.approx(1.0, abs=1e-9)


def test_kernel_matrix_is_psd():
    """A valid kernel matrix should be positive semidefinite."""
    X = np.random.default_rng(0).uniform(0, np.pi, size=(6, 2))
    K = quantum_kernel_matrix(X)
    eigs = np.linalg.eigvalsh(K)
    assert all(e >= -1e-9 for e in eigs), f"PSD violated: {eigs}"


def test_kernel_distinguishes_inputs():
    """For different x, y, the kernel should generally be < 1."""
    x = np.array([0.5, 0.5])
    y = np.array([2.0, 2.5])
    k = quantum_kernel_value(x, y)
    assert 0 <= k <= 1
    assert k < 1.0 - 1e-6


def test_kernel_matrix_cross_shape():
    """K(X, Y) has shape (n_x, n_y)."""
    X = np.random.default_rng(0).uniform(0, np.pi, size=(4, 2))
    Y = np.random.default_rng(1).uniform(0, np.pi, size=(3, 2))
    K = quantum_kernel_matrix(X, Y)
    assert K.shape == (4, 3)


def test_kernel_separates_distinct_classes():
    """A toy binary-class dataset: kernel should have higher mean intra-class
    similarity than inter-class similarity."""
    rng = np.random.default_rng(0)
    # Class A: centered at (0.5, 0.5); Class B: centered at (2.5, 2.5).
    class_a = np.array([[0.5, 0.5], [0.4, 0.6], [0.6, 0.5]])
    class_b = np.array([[2.5, 2.5], [2.6, 2.4], [2.4, 2.6]])
    X = np.vstack([class_a, class_b])
    K = quantum_kernel_matrix(X)
    # Block A-A (rows 0..2, cols 0..2), B-B (3..5, 3..5), A-B cross blocks.
    intra = (K[0:3, 0:3].mean() + K[3:6, 3:6].mean()) / 2
    inter = K[0:3, 3:6].mean()
    assert intra > inter
