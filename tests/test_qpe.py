import numpy as np
import pytest

from qbit_simulator.algorithms.qpe import (
    phase_estimation, phase_estimation_modexp,
    phase_estimation_modexp_marginal, estimate_phase_from_state,
)
from qbit_simulator.algorithms.shor import modular_multiplication_unitary


def test_qpe_phase_gate():
    """Eigenstate |1⟩ of P(2π/8) has eigenvalue e^{2πi·1/8}; QPE should return φ=1/8."""
    phi = 2 * np.pi / 8
    P = np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=np.complex128)
    eigenstate = np.array([0, 1], dtype=np.complex128)  # |1⟩
    qc = phase_estimation(P, eigenstate, n_counting=4)
    est = estimate_phase_from_state(qc, n_counting=4)
    assert est == pytest.approx(1 / 8, abs=1e-9)


@pytest.mark.parametrize("k", [1, 3, 5, 7])
def test_qpe_various_phases(k):
    phi = 2 * np.pi * k / 16
    P = np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=np.complex128)
    eigenstate = np.array([0, 1], dtype=np.complex128)
    qc = phase_estimation(P, eigenstate, n_counting=4)
    est = estimate_phase_from_state(qc, n_counting=4)
    assert est == pytest.approx(k / 16, abs=1e-9)


# ---- Fast-path equivalence: phase_estimation_modexp vs generic phase_estimation ----

@pytest.mark.parametrize("a,N,n_target,n_counting", [
    (2, 15, 4, 4),
    (7, 15, 4, 6),
    (4, 21, 5, 6),
    (2, 33, 6, 6),
])
def test_modexp_fastpath_matches_generic_state(a, N, n_target, n_counting):
    """The fast-path modexp QPE must produce a state-vector equal (up to a
    global phase) to the generic phase_estimation(U_a, |1⟩, t) path."""
    U_a = modular_multiplication_unitary(a, N, n_target)
    eig = np.zeros(2**n_target, dtype=np.complex128); eig[1] = 1.0
    qc_slow = phase_estimation(U_a, eig, n_counting)
    qc_fast = phase_estimation_modexp(a, N, n_target, n_counting)
    # Allow a global phase mismatch.
    inner = np.vdot(qc_slow.state, qc_fast.state)
    assert abs(abs(inner) - 1.0) < 1e-9


@pytest.mark.parametrize("a,N,k,t", [
    (2, 15, 4, 4),
    (7, 15, 4, 8),
    (4, 21, 5, 6),
    (2, 33, 6, 8),
    (5, 51, 6, 10),
])
def test_sparse_marginal_matches_dense_marginal(a, N, k, t):
    """The sparse-marginal path must produce the exact same P(c) the dense
    FFT path produces (within floating-point tolerance) — never materializing
    the dense 2^(t+k) state."""
    qc = phase_estimation_modexp(a, N, k, t)
    dense_marginal = qc.probabilities().reshape(1 << t, 1 << k).sum(axis=1)
    sparse_marginal = phase_estimation_modexp_marginal(a, N, k, t)
    assert np.allclose(dense_marginal, sparse_marginal, atol=1e-10)


def test_modexp_fastpath_marginal_peak_at_period_multiple():
    """For a=7, N=15, period r=4. QPE outcomes c/2^t should peak at multiples of 2^t/4."""
    n_target, n_counting = 4, 8
    qc = phase_estimation_modexp(7, 15, n_target, n_counting)
    probs = qc.probabilities()
    marginal = probs.reshape(1 << n_counting, 1 << n_target).sum(axis=1)
    # Peaks at c = 0, 64, 128, 192 (multiples of 2^8 / 4).
    peaks = np.argsort(marginal)[-4:]
    assert set(int(p) for p in peaks) == {0, 64, 128, 192}
