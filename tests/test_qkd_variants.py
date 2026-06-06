"""B92 and SARG04 tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.qkd_variants import (
    b92_run, sarg04_run, b92_security_threshold,
)


# ---- B92 ----

def test_b92_no_attack_zero_qber():
    rng = np.random.default_rng(0)
    r = b92_run(n_bits=5000, eve_attack=False, rng=rng)
    assert r["qber"] == 0.0


def test_b92_no_attack_sift_rate_around_25_pct():
    """B92 conclusive measurement rate is ~1/4 (Bob in Z half the time,
    50% of outcomes conclusive, etc.)."""
    rng = np.random.default_rng(0)
    r = b92_run(n_bits=5000, eve_attack=False, rng=rng)
    assert 0.10 < r["sift_rate"] < 0.40


def test_b92_eve_introduces_substantial_qber():
    """Intercept-resend by Eve should drive QBER well above the security
    threshold."""
    rng = np.random.default_rng(0)
    r = b92_run(n_bits=10000, eve_attack=True, rng=rng)
    assert r["qber"] > b92_security_threshold()


def test_b92_key_lengths_match():
    rng = np.random.default_rng(0)
    r = b92_run(n_bits=1000, rng=rng)
    assert len(r["alice_key"]) == len(r["bob_key"])
    assert len(r["alice_key"]) == r["n_sifted"]


def test_b92_zero_bits():
    rng = np.random.default_rng(0)
    r = b92_run(n_bits=0, rng=rng)
    assert r["n_sifted"] == 0
    assert r["qber"] == 0.0


def test_b92_threshold_is_reasonable():
    t = b92_security_threshold()
    assert 0.0 < t < 0.5


# ---- SARG04 ----

def test_sarg04_no_attack_zero_qber():
    rng = np.random.default_rng(0)
    r = sarg04_run(n_bits=5000, eve_attack=False, rng=rng)
    assert r["qber"] == 0.0


def test_sarg04_sift_rate_about_one_quarter():
    """SARG04 keeps ~1/4 of bits."""
    rng = np.random.default_rng(0)
    r = sarg04_run(n_bits=10000, eve_attack=False, rng=rng)
    assert 0.10 < r["sift_rate"] < 0.40


def test_sarg04_key_lengths_match():
    rng = np.random.default_rng(0)
    r = sarg04_run(n_bits=1000, rng=rng)
    assert len(r["alice_key"]) == len(r["bob_key"])


def test_sarg04_runs_with_eve():
    """Eavesdropping shouldn't crash the protocol."""
    rng = np.random.default_rng(0)
    r = sarg04_run(n_bits=1000, eve_attack=True, rng=rng)
    assert isinstance(r["qber"], float)


def test_sarg04_zero_bits():
    rng = np.random.default_rng(0)
    r = sarg04_run(n_bits=0, rng=rng)
    assert r["n_sifted"] == 0
