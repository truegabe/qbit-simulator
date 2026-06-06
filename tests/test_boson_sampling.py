"""Boson sampling tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.boson_sampling import (
    permanent, random_haar_unitary,
    boson_sampling_probability, boson_sampling_distribution,
    sample_boson_pattern, all_output_patterns,
)


# ---- permanent function ----

def test_permanent_of_identity():
    """Per(I_n) = 1 (only diagonal contributes)."""
    for n in (1, 2, 3, 4, 5):
        assert abs(permanent(np.eye(n, dtype=np.complex128)) - 1) < 1e-12


def test_permanent_of_2x2():
    """Per([[a,b],[c,d]]) = ad + bc."""
    a, b, c, d = 1.0, 2.0, 3.0, 4.0
    M = np.array([[a, b], [c, d]], dtype=np.complex128)
    assert abs(permanent(M) - (a*d + b*c)) < 1e-12


def test_permanent_of_3x3():
    """Per([[1,1,1],[1,1,1],[1,1,1]]) = 6 (number of permutations)."""
    M = np.ones((3, 3), dtype=np.complex128)
    assert abs(permanent(M) - 6) < 1e-12


def test_permanent_of_all_ones_n4():
    """Per(J_n) = n! (only permutations of n!)."""
    n = 4
    M = np.ones((n, n), dtype=np.complex128)
    assert abs(permanent(M) - 24) < 1e-10


# ---- Haar unitary ----

def test_haar_unitary_is_unitary():
    rng = np.random.default_rng(0)
    for n in (2, 4, 8):
        U = random_haar_unitary(n, rng)
        assert np.allclose(U @ U.conj().T, np.eye(n), atol=1e-12)


# ---- boson sampling probability ----

def test_distribution_normalized():
    """The distribution should sum to 1."""
    rng = np.random.default_rng(0)
    U = random_haar_unitary(4, rng)
    dist = boson_sampling_distribution(U, [0, 1])
    total = sum(dist.values())
    assert abs(total - 1.0) < 1e-9


def test_distribution_all_probabilities_nonnegative():
    rng = np.random.default_rng(0)
    U = random_haar_unitary(5, rng)
    dist = boson_sampling_distribution(U, [0, 1, 2])
    for p in dist.values():
        assert p >= -1e-12


def test_identity_unitary_concentrates_on_input():
    """If U = I, every photon stays in its input mode → only one pattern has
    probability 1 (the input pattern itself)."""
    U = np.eye(4, dtype=np.complex128)
    dist = boson_sampling_distribution(U, [0, 2])
    nonzero = {k: v for k, v in dist.items() if v > 1e-9}
    assert len(nonzero) == 1
    pattern = list(nonzero.keys())[0]
    assert sorted(pattern) == [0, 2]


# ---- sampling ----

def test_sample_returns_valid_pattern():
    rng = np.random.default_rng(0)
    U = random_haar_unitary(5, rng)
    pattern = sample_boson_pattern(U, [0, 1], rng=rng)
    assert len(pattern) == 2
    assert all(0 <= m < 5 for m in pattern)


def test_many_samples_match_distribution():
    """Empirical distribution from samples should converge to theoretical."""
    rng = np.random.default_rng(0)
    U = random_haar_unitary(3, rng)
    n_shots = 5000
    counts: dict = {}
    for _ in range(n_shots):
        p = sample_boson_pattern(U, [0, 1], rng=rng)
        counts[p] = counts.get(p, 0) + 1
    empirical = {k: v / n_shots for k, v in counts.items()}
    theoretical = boson_sampling_distribution(U, [0, 1])
    # Compare top-3 patterns.
    top_keys = sorted(theoretical.keys(),
                       key=lambda k: -theoretical[k])[:3]
    for k in top_keys:
        if k in empirical:
            assert abs(empirical[k] - theoretical[k]) < 0.05


# ---- pattern enumeration ----

def test_pattern_count_n2_m3():
    """Number of multisets of size n from m modes = C(m+n-1, n)."""
    patterns = all_output_patterns(2, 3)
    # C(4, 2) = 6.
    assert len(patterns) == 6
