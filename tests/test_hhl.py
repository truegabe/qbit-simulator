"""HHL algorithm tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.hhl import hhl


# ---- 2x2 diagonal systems ----

def test_hhl_diagonal_2x2():
    """Diagonal A = diag(1, 2) with |b⟩ = (|0⟩+|1⟩)/√2.
    Expected |x⟩ ∝ (1/1)|0⟩ + (1/2)|1⟩ = (2|0⟩+|1⟩)/√5."""
    A = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.complex128)
    b = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2)
    result = hhl(A, b, n_counting=6)
    expected = np.array([2.0, 1.0]) / np.sqrt(5)
    # Compare up to global phase.
    inner = abs(np.vdot(expected, result["x_quantum"]))
    assert inner > 0.95


def test_hhl_diagonal_2x2_unequal():
    """A = diag(1, 4), b = |0⟩+|1⟩. Expected |x⟩ ∝ (1 + |1⟩/4)/normalization."""
    A = np.array([[1.0, 0.0], [0.0, 4.0]], dtype=np.complex128)
    b = np.array([1.0, 1.0], dtype=np.complex128)
    result = hhl(A, b, n_counting=8)
    classical = np.linalg.solve(A, b)
    classical_norm = classical / np.linalg.norm(classical)
    inner = abs(np.vdot(classical_norm, result["x_quantum"]))
    assert inner > 0.95


def test_hhl_fidelity_increases_with_counting():
    """More clock qubits → higher fidelity (better eigenvalue resolution)."""
    A = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.complex128)
    b = np.array([0.6, 0.8], dtype=np.complex128)
    fid4 = hhl(A, b, n_counting=4)["fidelity"]
    fid8 = hhl(A, b, n_counting=8)["fidelity"]
    assert fid8 >= fid4 - 0.01   # allow slight tolerance


# ---- non-diagonal Hermitian 2x2 ----

def test_hhl_nondiagonal_2x2():
    """Random Hermitian A with nontrivial off-diagonal."""
    A = np.array([
        [2.0, 0.5],
        [0.5, 1.5],
    ], dtype=np.complex128)
    b = np.array([1.0, 0.0], dtype=np.complex128)
    result = hhl(A, b, n_counting=8)
    assert result["fidelity"] > 0.9


# ---- error checking ----

def test_hhl_rejects_non_hermitian():
    A = np.array([[1, 1], [0, 1]], dtype=np.complex128)
    b = np.array([1, 0], dtype=np.complex128)
    with pytest.raises(ValueError, match="Hermitian"):
        hhl(A, b)


def test_hhl_rejects_non_power_of_two():
    A = np.eye(3, dtype=np.complex128)
    b = np.array([1, 0, 0], dtype=np.complex128)
    with pytest.raises(ValueError, match="power of 2"):
        hhl(A, b)


def test_hhl_rejects_zero_b():
    A = np.eye(2, dtype=np.complex128)
    b = np.zeros(2, dtype=np.complex128)
    with pytest.raises(ValueError, match="nonzero"):
        hhl(A, b)


# ---- success probability is reported ----

def test_hhl_reports_success_probability():
    """Success probability should be a real number in (0, 1]."""
    A = np.array([[1.5, 0.0], [0.0, 2.0]], dtype=np.complex128)
    b = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2)
    result = hhl(A, b, n_counting=6)
    p = result["success_probability"]
    assert 0.0 < p <= 1.0
