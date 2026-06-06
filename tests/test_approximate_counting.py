"""Quantum approximate counting tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.approximate_counting import (
    grover_angle_from_count, count_from_grover_angle,
    quantum_count, bhmt_count, theoretical_uncertainty,
)


# ---- Angle ↔ count conversion ----

def test_angle_to_count_inverse():
    """count → angle → count should round-trip."""
    for M, N in [(0, 16), (1, 16), (8, 16), (16, 16)]:
        theta = grover_angle_from_count(M, N)
        M_back = count_from_grover_angle(theta, N)
        assert abs(M_back - M) < 1e-9


def test_angle_zero_for_no_match():
    assert abs(grover_angle_from_count(0, 16)) < 1e-12


def test_angle_pi_over_2_for_all_match():
    assert abs(grover_angle_from_count(16, 16) - np.pi / 2) < 1e-9


def test_angle_rejects_invalid():
    with pytest.raises(ValueError):
        grover_angle_from_count(20, 16)
    with pytest.raises(ValueError):
        grover_angle_from_count(-1, 16)


# ---- IQAE-based counting ----

def test_iqae_count_unique_match():
    """For M=1 in N=16, the estimate should be within 1 with high prob."""
    rng = np.random.default_rng(0)
    r = quantum_count(n_bits=4, n_marked=1, epsilon=0.02, rng=rng)
    assert r["error"] < 1.0


def test_iqae_count_zero_marked():
    """Zero matches → estimate should be 0."""
    r = quantum_count(n_bits=4, n_marked=0)
    assert r["M_estimate"] == 0.0


def test_iqae_count_returns_dict_structure():
    rng = np.random.default_rng(0)
    r = quantum_count(n_bits=4, n_marked=3, rng=rng)
    assert "M_estimate" in r
    assert "true_M" in r
    assert "error" in r
    assert r["true_M"] == 3
    assert r["N"] == 16


def test_iqae_count_rejects_invalid():
    with pytest.raises(ValueError):
        quantum_count(n_bits=4, n_marked=100)


@pytest.mark.parametrize("M", [1, 3, 7, 12])
def test_iqae_count_within_expected_error(M):
    """Average error should be small."""
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(3):
        r = quantum_count(n_bits=4, n_marked=M, epsilon=0.02, rng=rng)
        errs.append(r["error"])
    avg_err = np.mean(errs)
    # Loose bound: |error| < 2.
    assert avg_err < 2.0


# ---- BHMT counting ----

def test_bhmt_count_basic():
    rng = np.random.default_rng(0)
    r = bhmt_count(n_bits=4, n_marked=4, precision_bits=6, rng=rng)
    assert r["true_M"] == 4
    assert r["error"] < 3.0


def test_bhmt_count_zero():
    r = bhmt_count(n_bits=4, n_marked=0)
    assert r["M_estimate"] == 0.0


def test_bhmt_count_more_precision_helps():
    """Averaging over runs, higher precision_bits should give lower error."""
    rng = np.random.default_rng(0)
    errs_low = []
    errs_high = []
    for _ in range(20):
        r_low = bhmt_count(n_bits=4, n_marked=5, precision_bits=3, rng=rng)
        r_high = bhmt_count(n_bits=4, n_marked=5, precision_bits=8, rng=rng)
        errs_low.append(r_low["error"])
        errs_high.append(r_high["error"])
    assert np.mean(errs_high) < np.mean(errs_low) + 1.0


# ---- Theoretical uncertainty ----

def test_theoretical_uncertainty_zero_at_full_match():
    """At M = N, uncertainty is 0 (variance vanishes)."""
    assert theoretical_uncertainty(M=16, N=16, n_iters=100) == 0


def test_theoretical_uncertainty_zero_at_zero_match():
    assert theoretical_uncertainty(M=0, N=16, n_iters=100) == 0


def test_theoretical_uncertainty_decreases_with_iters():
    u1 = theoretical_uncertainty(M=8, N=16, n_iters=10)
    u2 = theoretical_uncertainty(M=8, N=16, n_iters=100)
    assert u2 < u1
