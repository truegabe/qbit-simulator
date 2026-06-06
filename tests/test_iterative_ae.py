"""Iterative / Maximum-Likelihood Amplitude Estimation tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.iterative_ae import (
    grover_measurement_probability, sample_grover_outcomes,
    mlae_estimate, iqae_estimate,
)


# ---- Grover measurement probability ----

def test_grover_probability_at_k_zero():
    """At k=0 (no Grover iterations), P(good) = sin²(θ)."""
    theta = 0.3
    p = grover_measurement_probability(theta, k=0)
    assert abs(p - np.sin(theta) ** 2) < 1e-12


def test_grover_probability_at_k1():
    """At k=1, P(good) = sin²(3θ)."""
    theta = 0.2
    p = grover_measurement_probability(theta, k=1)
    assert abs(p - np.sin(3 * theta) ** 2) < 1e-12


# ---- MLAE recovery ----

@pytest.mark.parametrize("theta_true", [0.1, 0.3, 0.5, 0.7, 1.0])
def test_mlae_recovers_theta_accurately(theta_true):
    """MLAE should recover θ to better than ε ≈ 1/(max k)."""
    rng = np.random.default_rng(int(theta_true * 100))
    depths = [0, 1, 2, 4, 8, 16]
    r = mlae_estimate(theta_true, depths, n_shots_per_depth=500, rng=rng)
    # Resolution at max k=16 is roughly 1/16 = 0.06.
    assert abs(r["theta_estimate"] - theta_true) < 0.1


def test_mlae_amplitude_estimate():
    """For θ = π/6 → a = sin²(π/6) = 0.25."""
    theta = np.pi / 6
    rng = np.random.default_rng(0)
    depths = [0, 1, 2, 4, 8]
    r = mlae_estimate(theta, depths, n_shots_per_depth=500, rng=rng)
    assert abs(r["amplitude_estimate"] - 0.25) < 0.05


def test_mlae_returns_counts():
    rng = np.random.default_rng(0)
    r = mlae_estimate(0.3, [0, 1, 2], n_shots_per_depth=100, rng=rng)
    assert len(r["counts"]) == 3
    for k, m in r["counts"]:
        assert 0 <= m <= 100


# ---- IQAE recovery ----

@pytest.mark.parametrize("theta_true", [0.2, 0.5, 0.9, 1.2])
def test_iqae_recovers_theta(theta_true):
    rng = np.random.default_rng(int(theta_true * 100))
    r = iqae_estimate(theta_true, epsilon=0.02, n_shots_per_round=200,
                       max_iter=30, rng=rng)
    assert abs(r["theta_estimate"] - theta_true) < 0.1


def test_iqae_interval_narrows():
    """The output interval should be narrow."""
    rng = np.random.default_rng(0)
    r = iqae_estimate(0.5, epsilon=0.05, n_shots_per_round=100, rng=rng)
    lo, hi = r["theta_interval"]
    # Should have narrowed below the initial π/2.
    assert hi - lo < np.pi / 2
