"""Tests for the noise channels and single-qubit randomized benchmarking."""

import numpy as np
import pytest

from qbit_simulator.noise import (
    bit_flip_kraus, phase_flip_kraus, depolarizing_kraus,
    amplitude_damping_kraus, phase_damping_kraus, thermal_relaxation_kraus,
)
from qbit_simulator.algorithms.randomized_benchmarking import (
    _CLIFFORDS, random_single_qubit_clifford,
    run_rb_sequence, randomized_benchmarking,
)


# ---- Kraus operator validity ----

def _is_trace_preserving(kraus: list[np.ndarray], tol: float = 1e-9) -> bool:
    """Σ K_i† K_i should equal I (TPCP condition)."""
    s = sum(K.conj().T @ K for K in kraus)
    return np.allclose(s, np.eye(s.shape[0]), atol=tol)


def test_bit_flip_is_trace_preserving():
    for p in (0.0, 0.1, 0.5, 1.0):
        assert _is_trace_preserving(bit_flip_kraus(p))


def test_phase_flip_is_trace_preserving():
    for p in (0.0, 0.1, 0.5, 1.0):
        assert _is_trace_preserving(phase_flip_kraus(p))


def test_depolarizing_is_trace_preserving():
    for p in (0.0, 0.1, 0.5, 1.0):
        assert _is_trace_preserving(depolarizing_kraus(p))


def test_amplitude_damping_is_trace_preserving():
    for gamma in (0.0, 0.1, 0.5, 0.99):
        assert _is_trace_preserving(amplitude_damping_kraus(gamma))


def test_phase_damping_is_trace_preserving():
    for lam in (0.0, 0.1, 0.5, 0.99):
        assert _is_trace_preserving(phase_damping_kraus(lam))


def test_thermal_relaxation_is_trace_preserving():
    """T1/T2 channel is the composition of amp damping + phase damping."""
    for t1, t2 in [(100, 50), (100, 100), (1000, 1000)]:
        kraus = thermal_relaxation_kraus(t1=t1, t2=t2, gate_time=10)
        assert _is_trace_preserving(kraus)


def test_thermal_relaxation_rejects_t2_gt_2t1():
    with pytest.raises(ValueError):
        thermal_relaxation_kraus(t1=100, t2=300, gate_time=10)


# ---- Clifford group enumeration ----

def test_clifford_group_has_24_elements():
    assert len(_CLIFFORDS) == 24


def test_all_cliffords_are_unitary():
    for U in _CLIFFORDS:
        assert np.allclose(U @ U.conj().T, np.eye(2), atol=1e-9)


def test_random_clifford_returns_2x2_unitary():
    rng = np.random.default_rng(0)
    for _ in range(20):
        U = random_single_qubit_clifford(rng)
        assert U.shape == (2, 2)
        assert np.allclose(U @ U.conj().T, np.eye(2), atol=1e-9)


# ---- noise-free RB returns to |0⟩ ----

def test_noise_free_rb_always_returns_zero():
    """Without noise, the recovery Clifford should restore |0⟩ exactly."""
    rng = np.random.default_rng(42)
    for _ in range(30):
        outcome = run_rb_sequence(n_gates=20, kraus_per_gate=None, rng=rng)
        assert outcome == 0


@pytest.mark.parametrize("n_gates", [1, 5, 20, 50])
def test_noise_free_rb_at_various_lengths(n_gates):
    rng = np.random.default_rng(0)
    for _ in range(10):
        assert run_rb_sequence(n_gates, kraus_per_gate=None, rng=rng) == 0


# ---- RB recovers the noise parameter (qualitative) ----

def test_rb_decay_detects_depolarizing_noise():
    """A non-trivial depolarizing channel should produce a measurable decay."""
    rng = np.random.default_rng(0)
    p_noise = 0.05
    kraus = depolarizing_kraus(p_noise)
    lengths = [1, 5, 10, 20, 50]
    result = randomized_benchmarking(
        lengths, kraus_per_gate=kraus, n_trials=200, rng=rng,
    )
    # With p_noise = 5%, RB-fit p should be ≤ 1 (with some statistical noise).
    assert result.p_fit < 1.0
    # And r should be positive.
    assert result.average_gate_error > 0.0


def test_rb_decay_with_no_noise_gives_p_near_1():
    """No noise → p ≈ 1, r ≈ 0."""
    rng = np.random.default_rng(0)
    lengths = [1, 5, 10, 20]
    result = randomized_benchmarking(
        lengths, kraus_per_gate=None, n_trials=50, rng=rng,
    )
    assert result.p_fit > 0.95
    assert result.average_gate_error < 0.05


def test_rb_decay_stronger_noise_gives_more_decay():
    """A stronger noise channel → smaller p, larger r."""
    rng = np.random.default_rng(0)
    lengths = [1, 5, 10, 20, 50]
    r_small = randomized_benchmarking(
        lengths, kraus_per_gate=depolarizing_kraus(0.01),
        n_trials=300, rng=rng,
    )
    r_large = randomized_benchmarking(
        lengths, kraus_per_gate=depolarizing_kraus(0.10),
        n_trials=300, rng=np.random.default_rng(0),
    )
    assert r_large.p_fit < r_small.p_fit
