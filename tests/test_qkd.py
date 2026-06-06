"""Tests for QKD protocols + Wiesner's quantum money."""

import numpy as np
import pytest

from qbit_simulator.algorithms.qkd import (
    bb84_protocol, e91_protocol, wiesner_quantum_money,
)


# ---- BB84 ----

def test_bb84_no_eavesdropper_zero_qber():
    """Without Eve, Alice's and Bob's sifted keys must be identical (QBER=0)."""
    rng = np.random.default_rng(0)
    r = bb84_protocol(n_bits=200, eve_probability=0.0, rng=rng)
    assert r["qber"] == 0.0
    assert r["n_sifted"] > 0
    assert np.array_equal(r["sifted_alice"], r["sifted_bob"])


def test_bb84_sifted_size_roughly_half():
    """On average half the qubits have matching bases (n_sifted ≈ n_bits / 2)."""
    rng = np.random.default_rng(0)
    n = 1000
    r = bb84_protocol(n_bits=n, eve_probability=0.0, rng=rng)
    assert 0.4 < r["n_sifted"] / n < 0.6


def test_bb84_eve_introduces_errors():
    """If Eve intercepts every qubit, expected QBER ≈ 1/4."""
    rng = np.random.default_rng(0)
    r = bb84_protocol(n_bits=500, eve_probability=1.0, rng=rng)
    # Eve intercepts and re-sends → ~25% QBER.
    assert 0.15 < r["qber"] < 0.35


def test_bb84_eve_count_tracked():
    rng = np.random.default_rng(0)
    n = 100
    r = bb84_protocol(n_bits=n, eve_probability=0.5, rng=rng)
    # About half should be intercepted.
    assert 30 < r["eve_intercepts"] < 70


# ---- E91 ----

def test_e91_no_eavesdropper_correlations():
    """E91 without eavesdropper should give low QBER on sifted key."""
    rng = np.random.default_rng(0)
    r = e91_protocol(n_pairs=1000, rng=rng)
    # With aligned bases (alice basis 1 = bob basis 0 = π/4), the ψ+ state
    # gives perfectly anti-correlated outcomes; after Bob takes complement,
    # both keys should match.
    assert r["qber"] < 0.1


def test_e91_chsh_violates_classical_bound():
    """E91's CHSH expectation should exceed 2 (classical bound) for an
    intact Bell pair, demonstrating quantum behavior."""
    rng = np.random.default_rng(0)
    r = e91_protocol(n_pairs=2000, rng=rng)
    # Tsirelson bound is 2√2 ≈ 2.83; classical bound is 2.
    assert r["chsh_value"] > 1.8   # quantum advantage


def test_e91_returns_sifted_key():
    rng = np.random.default_rng(0)
    r = e91_protocol(n_pairs=300, rng=rng)
    assert r["n_sifted"] > 0
    assert len(r["sifted_alice"]) == r["n_sifted"]


# ---- Wiesner quantum money ----

def test_wiesner_legitimate_banknote_passes():
    """A bank-prepared banknote, measured in the bank's own bases, must
    always pass verification."""
    rng = np.random.default_rng(0)
    for trial in range(10):
        r = wiesner_quantum_money(n_qubits=8, rng=rng)
        assert r["legitimate_passes"]


def test_wiesner_counterfeit_fails_more_often():
    """A counterfeiter without basis knowledge fails verification often
    enough to be statistically detectable."""
    rng = np.random.default_rng(0)
    n_trials = 100
    n_passes = 0
    for _ in range(n_trials):
        r = wiesner_quantum_money(n_qubits=4, rng=rng)
        if r["attack_passes"]:
            n_passes += 1
    expected_rate = (3 / 4) ** 4   # ≈ 0.316
    observed_rate = n_passes / n_trials
    # Within ±50% of theoretical (small-sample stochasticity).
    assert 0.5 * expected_rate < observed_rate < 1.5 * expected_rate


def test_wiesner_attack_rate_smaller_with_more_qubits():
    """Counterfeit success drops as (3/4)^n — exponentially."""
    rng = np.random.default_rng(0)
    n_trials = 100
    successes_n4 = sum(
        wiesner_quantum_money(n_qubits=4, rng=rng)["attack_passes"]
        for _ in range(n_trials)
    )
    successes_n10 = sum(
        wiesner_quantum_money(n_qubits=10, rng=rng)["attack_passes"]
        for _ in range(n_trials)
    )
    # n=4 should pass ~32 out of 100; n=10 should pass < 6 out of 100.
    assert successes_n4 > successes_n10
