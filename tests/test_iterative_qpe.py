"""Iterative QPE tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.iterative_qpe import iterative_qpe


def test_iqpe_eighth_phase():
    """U = P(2π/8) has eigenstate |1⟩ with phase φ = 1/8."""
    phi_true = 1 / 8
    U = np.array([[1, 0], [0, np.exp(2j * np.pi * phi_true)]], dtype=np.complex128)
    eig = np.array([0, 1], dtype=np.complex128)
    result = iterative_qpe(U, eig, n_bits=6, rng=np.random.default_rng(0))
    # Should recover φ = 1/8 = 0.125 within 1/2^6 = ~0.016.
    assert abs(result["phase"] - phi_true) < 1 / 64


@pytest.mark.parametrize("k", [1, 3, 5, 7])
def test_iqpe_various_phases(k):
    """U = P(2π · k/16), eigenstate |1⟩ → φ = k/16."""
    phi_true = k / 16
    U = np.array([[1, 0], [0, np.exp(2j * np.pi * phi_true)]], dtype=np.complex128)
    eig = np.array([0, 1], dtype=np.complex128)
    result = iterative_qpe(U, eig, n_bits=8, rng=np.random.default_rng(k))
    assert abs(result["phase"] - phi_true) < 1 / 128


def test_iqpe_zero_phase():
    """φ = 0 should be recoverable."""
    U = np.eye(2, dtype=np.complex128)
    eig = np.array([1, 0], dtype=np.complex128)
    result = iterative_qpe(U, eig, n_bits=5, rng=np.random.default_rng(0))
    assert result["phase"] == 0.0


def test_iqpe_returns_correct_number_of_bits():
    U = np.array([[1, 0], [0, np.exp(1j * np.pi / 8)]], dtype=np.complex128)
    eig = np.array([0, 1], dtype=np.complex128)
    for n in (3, 5, 8, 10):
        result = iterative_qpe(U, eig, n_bits=n, rng=np.random.default_rng(0))
        assert len(result["bits"]) == n


def test_iqpe_two_qubit_unitary():
    """Phase estimation on a 2-qubit unitary."""
    # Diagonal 2-qubit unitary with phase 3/16 on |11⟩.
    phi_true = 3 / 16
    U = np.eye(4, dtype=np.complex128)
    U[3, 3] = np.exp(2j * np.pi * phi_true)
    eig = np.zeros(4, dtype=np.complex128); eig[3] = 1
    result = iterative_qpe(U, eig, n_bits=8, rng=np.random.default_rng(0))
    assert abs(result["phase"] - phi_true) < 1 / 128
