"""Gottesman-Chuang quantum digital signature tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.quantum_signatures import (
    gc_setup, gc_sign, gc_verify,
    gc_forge_attempt, gc_forge_failure_probability,
    gc_correctness_demo,
)


# ---- Setup ----

def test_setup_creates_correct_number_of_copies():
    rng = np.random.default_rng(0)
    kp = gc_setup(M=42, rng=rng)
    assert kp.M == 42
    assert len(kp.public_copies) == 42


def test_setup_with_explicit_bit():
    kp = gc_setup(M=10, secret_bit=1)
    assert kp.secret_bit == 1


def test_setup_rejects_bad_bit():
    with pytest.raises(ValueError):
        gc_setup(M=10, secret_bit=2)


def test_public_copies_are_normalized():
    kp = gc_setup(M=10, secret_bit=0)
    for psi in kp.public_copies:
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-12


def test_public_copies_match_secret_bit():
    kp0 = gc_setup(M=5, secret_bit=0)
    kp1 = gc_setup(M=5, secret_bit=1)
    # Bit-0 copies are |0⟩.
    for psi in kp0.public_copies:
        assert abs(psi[0] - 1) < 1e-12
    # Bit-1 copies are |+⟩.
    for psi in kp1.public_copies:
        assert abs(psi[0] - 1 / np.sqrt(2)) < 1e-12


# ---- Signing ----

def test_sign_returns_bit():
    kp = gc_setup(M=10, secret_bit=0)
    sig = gc_sign(0, kp)
    assert sig == 0


# ---- Verification ----

def test_honest_verification_always_passes():
    rng = np.random.default_rng(0)
    for _ in range(50):
        r = gc_correctness_demo(M=30, threshold=0.1, rng=rng)
        assert r["accept"]


def test_honest_zero_mismatches_for_bit_zero():
    """Verifying |0⟩ copies in Z basis gives 0 always."""
    rng = np.random.default_rng(0)
    kp = gc_setup(M=20, secret_bit=0)
    r = gc_verify(0, kp.public_copies, rng=rng)
    assert r["n_mismatches"] == 0


def test_honest_zero_mismatches_for_bit_one():
    """Verifying |+⟩ copies in X basis gives 0 always."""
    rng = np.random.default_rng(0)
    kp = gc_setup(M=20, secret_bit=1)
    r = gc_verify(1, kp.public_copies, rng=rng)
    assert r["n_mismatches"] == 0


# ---- Forgery ----

def test_forgery_rate_at_50_percent():
    """A forger claiming the wrong bit gets ~50% mismatches."""
    rng = np.random.default_rng(0)
    r = gc_forge_attempt(claimed_bit=1, true_bit=0, M=200, threshold=1.0, rng=rng)
    # threshold=1.0 means accept; but check the mismatch rate.
    assert abs(r["mismatch_rate"] - 0.5) < 0.1


def test_forgery_rarely_accepted_with_large_M():
    """With M=50 and 10% threshold, forgery should almost never succeed."""
    rng = np.random.default_rng(0)
    n_succeed = 0
    for _ in range(200):
        r = gc_forge_attempt(claimed_bit=1, true_bit=0, M=50, threshold=0.1, rng=rng)
        if r["accept"]:
            n_succeed += 1
    assert n_succeed < 5


def test_forge_failure_probability_increases_with_M():
    """Larger M → higher rejection probability."""
    p_5 = gc_forge_failure_probability(M=5, threshold=0.1)
    p_50 = gc_forge_failure_probability(M=50, threshold=0.1)
    p_500 = gc_forge_failure_probability(M=500, threshold=0.1)
    assert p_5 < p_50 < p_500
    assert p_500 > 0.999999


def test_forge_failure_probability_bounded():
    assert 0 <= gc_forge_failure_probability(20, 0.1) <= 1


# ---- Integration ----

def test_full_protocol_signer_to_verifier():
    """End-to-end: setup → sign → verify."""
    rng = np.random.default_rng(0)
    kp = gc_setup(M=30, secret_bit=0, rng=rng)
    sig = gc_sign(kp.secret_bit, kp)
    r = gc_verify(sig, kp.public_copies, rng=rng)
    assert r["accept"]
