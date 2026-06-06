"""Quantum coin flipping (BCJL-flavored) tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.coin_flipping import (
    bcjl_state, bcjl_verify_probability,
    bcjl_round, bcjl_simulate, bcjl_max_bias,
)


# ---- Commit states ----

def test_commit_states_normalized():
    assert abs(np.linalg.norm(bcjl_state(0)) - 1.0) < 1e-12
    assert abs(np.linalg.norm(bcjl_state(1)) - 1.0) < 1e-12


def test_commit_states_overlap():
    """⟨0|+⟩ = 1/√2."""
    overlap = abs(np.vdot(bcjl_state(0), bcjl_state(1)))
    assert abs(overlap - 1 / np.sqrt(2)) < 1e-12


def test_commit_states_rejects_bad_bit():
    with pytest.raises(ValueError):
        bcjl_state(2)


# ---- Verification ----

def test_verification_honest_alice_always_passes():
    """If Alice sends |ψ_a⟩ and claims a, verification succeeds with p=1."""
    for a in (0, 1):
        psi = bcjl_state(a)
        assert abs(bcjl_verify_probability(psi, a) - 1.0) < 1e-12


def test_verification_cheat_state_pass_prob():
    """The optimal cheat state (|ψ_0⟩+|ψ_1⟩)/N passes with prob (1+overlap)/2."""
    cheat = bcjl_state(0) + bcjl_state(1)
    cheat = cheat / np.linalg.norm(cheat)
    p0 = bcjl_verify_probability(cheat, 0)
    p1 = bcjl_verify_probability(cheat, 1)
    expected = (1 + 1 / np.sqrt(2)) / 2
    assert abs(p0 - expected) < 1e-9
    assert abs(p1 - expected) < 1e-9


def test_verification_with_density_matrix():
    rho = np.outer(bcjl_state(0), bcjl_state(0).conj())
    assert abs(bcjl_verify_probability(rho, 0) - 1.0) < 1e-12


# ---- One round ----

def test_honest_round_completes():
    rng = np.random.default_rng(0)
    r = bcjl_round(rng)
    assert not r["abort"]
    assert r["outcome"] in (0, 1)
    assert r["cheating"] == "none"


def test_alice_cheat_sometimes_caught():
    """Cheating Alice fails verification ~14.6% of the time."""
    rng = np.random.default_rng(0)
    n_aborts = sum(bcjl_round(rng, alice_cheats=True, alice_target=0)["abort"]
                    for _ in range(5000))
    abort_rate = n_aborts / 5000
    expected_abort = 1 - (1 + 1 / np.sqrt(2)) / 2   # ≈ 0.146
    assert abs(abort_rate - expected_abort) < 0.02


# ---- Multi-round statistics ----

def test_honest_statistics_unbiased():
    rng = np.random.default_rng(0)
    r = bcjl_simulate(20_000, rng=rng)
    assert abs(r["p_zero"] - 0.5) < 0.03
    assert r["abort_rate"] == 0.0


def test_cheating_alice_biases_outcome():
    """When Alice cheats toward 0 and passes verification, outcome is 0."""
    rng = np.random.default_rng(0)
    r = bcjl_simulate(20_000, alice_cheats=True, alice_target=0, rng=rng)
    # Conditioned on non-aborted runs: Alice's bias should be very large.
    assert r["p_zero"] > 0.95
    # Some aborts.
    assert r["abort_rate"] > 0.10


def test_cheating_alice_toward_one():
    rng = np.random.default_rng(0)
    r = bcjl_simulate(20_000, alice_cheats=True, alice_target=1, rng=rng)
    assert r["p_one"] > 0.95


# ---- Theoretical bound ----

def test_max_bias_value():
    """Max bias for BCJL with {|0⟩, |+⟩}: 1/√2 / 2 = 1/(2√2)."""
    expected = (1 / np.sqrt(2)) / 2
    assert abs(bcjl_max_bias() - expected) < 1e-9


def test_max_bias_positive():
    assert 0 < bcjl_max_bias() < 0.5
