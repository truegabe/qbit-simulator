"""Quantum Amplitude Estimation tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.amplitude_estimation import (
    amplitude_estimation, grover_operator, make_ry_test_unitary,
)


# ---- grover operator sanity ----

def test_grover_operator_is_unitary():
    """Q = -A S_0 A^-1 S_ψ must be unitary for any unitary A."""
    theta = 0.7
    A = make_ry_test_unitary(theta)
    Q = grover_operator(A)
    assert np.allclose(Q.conj().T @ Q, np.eye(2), atol=1e-12)


def test_grover_eigenvalues_match_2theta():
    """Q has eigenvalues e^{±2iθ}."""
    for theta in (0.3, 0.7, 1.1, np.pi / 4):
        A = make_ry_test_unitary(theta)
        Q = grover_operator(A)
        eigvals = np.linalg.eigvals(Q)
        expected = {np.exp(2j * theta), np.exp(-2j * theta)}
        # Check that each computed eigenvalue is close to one of the expected.
        for ev in eigvals:
            assert any(abs(ev - e) < 1e-9 for e in expected), \
                f"eigenvalue {ev} not near {expected} for θ={theta}"


# ---- amplitude estimation correctness ----

@pytest.mark.parametrize("theta,expected_a", [
    (np.pi / 6, 0.25),     # sin²(π/6) = 1/4
    (np.pi / 4, 0.5),      # sin²(π/4) = 1/2
    (np.pi / 3, 0.75),     # sin²(π/3) = 3/4
])
def test_amplitude_estimation_recovers_known_a(theta, expected_a):
    """For A = Ry(2θ), QAE should return a close to sin²(θ)."""
    A = make_ry_test_unitary(theta)
    result = amplitude_estimation(A, n_counting=8)
    # Resolution is ~ 1/2^8 in phase, which translates to ~ π/256 in θ.
    # In amplitude, derivative is sin(2θ), so error is at most sin(2θ) · π/256.
    max_err = abs(np.sin(2 * theta)) * np.pi / 256 + 1e-3
    assert abs(result["amplitude"] - expected_a) < max_err


def test_amplitude_estimation_high_precision():
    """With n_counting=12, the answer should be very close."""
    theta = np.pi / 5
    A = make_ry_test_unitary(theta)
    result = amplitude_estimation(A, n_counting=12)
    expected_a = np.sin(theta) ** 2
    assert abs(result["amplitude"] - expected_a) < 1e-3


def test_amplitude_estimation_a_equals_half():
    """When θ = π/4 the amplitude is exactly 1/2."""
    A = make_ry_test_unitary(np.pi / 4)
    result = amplitude_estimation(A, n_counting=8)
    assert abs(result["amplitude"] - 0.5) < 1e-6


def test_amplitude_estimation_marginal_has_two_peaks():
    """Q has two eigenvalues with mirror phases; QPE produces two peaks."""
    A = make_ry_test_unitary(np.pi / 5)
    result = amplitude_estimation(A, n_counting=8)
    marginal = result["counting_marginal"]
    # The two peaks should sum to ≥ 0.8 of total probability.
    top_two_mass = np.sort(marginal)[-2:].sum()
    assert top_two_mass > 0.8


# ---- multi-qubit state prep ----

def test_amplitude_estimation_for_n2_state():
    """Use a 2-qubit state-prep that creates a known superposition."""
    # Construct A = (Ry(2θ) ⊗ I) — first qubit is data, second is flag.
    # Wait — convention: LAST qubit is the flag. Let's make A such that
    # A|00⟩ = cos(θ)|00⟩ + sin(θ)|01⟩ (flag = last qubit).
    theta = 0.6
    c = np.cos(theta); s = np.sin(theta)
    # Ry-like 4x4 acting on the last qubit, treating first qubit as identity.
    A = np.array([
        [c, -s, 0,  0],
        [s,  c, 0,  0],
        [0,  0, c, -s],
        [0,  0, s,  c],
    ], dtype=np.complex128)
    result = amplitude_estimation(A, n_counting=8)
    expected_a = np.sin(theta) ** 2
    # Bigger error budget for the 2-qubit version because of finite resolution.
    assert abs(result["amplitude"] - expected_a) < 0.05


# ---- edge cases ----

def test_amplitude_estimation_a_zero():
    """If A|0⟩ = |0⟩ (no good states), a = 0."""
    A = np.eye(2, dtype=np.complex128)
    result = amplitude_estimation(A, n_counting=6)
    assert result["amplitude"] < 0.05


def test_amplitude_estimation_a_one():
    """If A|0⟩ = |1⟩ (all amplitude on good state), a = 1."""
    A = np.array([[0, 1], [1, 0]], dtype=np.complex128)  # X gate
    result = amplitude_estimation(A, n_counting=6)
    assert result["amplitude"] > 0.95
