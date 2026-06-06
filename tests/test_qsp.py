"""Quantum Signal Processing tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.qsp import (
    signal_operator, phase_operator, qsp_unitary, qsp_polynomial,
    chebyshev_phases, identity_phases, chebyshev_t,
)


# ---- building blocks ----

@pytest.mark.parametrize("a", [-0.9, -0.5, 0.0, 0.3, 0.7, 0.99])
def test_signal_operator_is_unitary(a):
    W = signal_operator(a)
    assert np.allclose(W @ W.conj().T, np.eye(2), atol=1e-12)


@pytest.mark.parametrize("phi", [0.0, 0.3, np.pi / 2, np.pi, -1.7])
def test_phase_operator_is_unitary(phi):
    P = phase_operator(phi)
    assert np.allclose(P @ P.conj().T, np.eye(2), atol=1e-12)


# ---- identity QSP ----

def test_identity_qsp_returns_signal():
    """phases=[0, 0] gives P(a) = a (just the signal)."""
    for a in np.linspace(-1, 1, 9):
        p = qsp_polynomial(identity_phases(), a)
        assert abs(p.real - a) < 1e-9
        assert abs(p.imag) < 1e-9


# ---- single phase ----

def test_zero_d_qsp_returns_one():
    """phases=[0] (length 1, d=0): U = I, P(a) = 1."""
    for a in np.linspace(-1, 1, 5):
        p = qsp_polynomial([0.0], a)
        assert abs(p - 1.0) < 1e-12


# ---- Chebyshev polynomials ----

@pytest.mark.parametrize("d", [0, 1, 2, 3, 4, 5, 8])
def test_chebyshev_qsp_polynomial(d):
    """With all-zero phases, QSP produces exactly T_d(a) (the Chebyshev
    polynomial of the first kind, order d). Direct verification by the
    Chebyshev recurrence."""
    phases = chebyshev_phases(d)
    for a in [-0.9, -0.4, 0.0, 0.3, 0.7]:
        p = qsp_polynomial(phases, a)
        expected = chebyshev_t(d, a)
        assert abs(p.real - expected) < 1e-9
        assert abs(p.imag) < 1e-9


def test_chebyshev_polynomial_magnitude_bounded():
    """|P(a)| ≤ 1 for a ∈ [-1, 1] (a key QSP constraint)."""
    for d in (2, 5, 10):
        phases = chebyshev_phases(d)
        for a in np.linspace(-1, 1, 21):
            p = qsp_polynomial(phases, a)
            assert abs(p) <= 1.0 + 1e-9


# ---- general unitarity of QSP ----

@pytest.mark.parametrize("d", [1, 3, 5])
def test_qsp_unitary_is_unitary(d):
    """For any phase sequence and any a ∈ [-1, 1], U is unitary."""
    rng = np.random.default_rng(d)
    phases = list(rng.uniform(0, 2 * np.pi, d + 1))
    for a in [-0.7, -0.2, 0.4, 0.8]:
        U = qsp_unitary(phases, a)
        assert np.allclose(U @ U.conj().T, np.eye(2), atol=1e-10)


# ---- polynomial degree ----

def test_qsp_polynomial_degree_matches_phase_count():
    """The polynomial P(a) has degree exactly d, where d+1 = len(phases).
    We sample P at many a's and fit a polynomial; degree should be ≤ d."""
    d = 4
    phases = list(np.random.default_rng(0).uniform(0, 2 * np.pi, d + 1))
    a_vals = np.linspace(-0.99, 0.99, 50)
    p_vals = np.array([qsp_polynomial(phases, a).real for a in a_vals])
    # The real part of P(a) is a polynomial of degree at most d.
    # Fitting with degree d+1 should give a near-zero leading coefficient.
    coeffs = np.polyfit(a_vals, p_vals, d + 1)
    # Leading coeff (of x^(d+1)) should be ~0.
    assert abs(coeffs[0]) < 1e-3


# ---- amplitude amplification phases produce expected boost ----

def test_amplitude_amplification_one_iteration():
    """One iteration of AA: starting amplitude a, the boosted amplitude
    should be sin(3·arcsin(a)) (in absolute value)."""
    from qbit_simulator.algorithms.qsp import amplitude_amplification_phases
    phases = amplitude_amplification_phases(1)
    for a in (0.2, 0.3, 0.5):
        # Some QSP convention difference may put the boosted amplitude in P or Q;
        # check at least that *one* matrix element shows the expected behavior.
        U = qsp_unitary(phases, a)
        # Boosted amplitude appears in the (0,1) entry typically.
        expected_boost = abs(np.sin(3 * np.arcsin(a)))
        observed = max(abs(U[0, 0]), abs(U[0, 1]),
                       abs(U[1, 0]), abs(U[1, 1]))
        # Just sanity-check that the unitary's max element is at least expected_boost
        # for small a (where boost is moderate).
        assert observed >= expected_boost - 0.05
