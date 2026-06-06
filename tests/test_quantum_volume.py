"""Quantum Volume benchmark tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.quantum_volume import (
    _haar_u4, _apply_2q_on_pair, _apply_depolarizing,
    model_circuit, heavy_output_set, heavy_output_probability,
    quantum_volume_trial, quantum_volume_estimate, find_quantum_volume,
)


# ---- Haar U(4) ----

def test_haar_u4_is_unitary():
    rng = np.random.default_rng(0)
    for _ in range(5):
        U = _haar_u4(rng)
        assert np.allclose(U @ U.conj().T, np.eye(4), atol=1e-9)


def test_haar_u4_is_4x4():
    U = _haar_u4(np.random.default_rng(0))
    assert U.shape == (4, 4)


# ---- 2-qubit gate application ----

def test_apply_2q_preserves_norm():
    rng = np.random.default_rng(0)
    psi = rng.normal(size=8) + 1j * rng.normal(size=8)
    psi /= np.linalg.norm(psi)
    U = _haar_u4(rng)
    out = _apply_2q_on_pair(psi, U, q0=0, q1=1, n=3)
    assert abs(np.linalg.norm(out) - 1.0) < 1e-10


def test_apply_2q_correct_position():
    """U on (q0=0, q1=2) of a 3-qubit state should leave q1 untouched."""
    rng = np.random.default_rng(0)
    # Use CNOT(0, 2) as a test.
    CNOT = np.array([
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0],
    ], dtype=complex)
    psi = np.zeros(8, dtype=complex); psi[4] = 1.0   # |100⟩
    out = _apply_2q_on_pair(psi, CNOT, q0=0, q1=2, n=3)
    # CNOT(0,2): control=q0=1 → flip q2 → |101⟩ = idx 5.
    assert abs(abs(out[5]) - 1.0) < 1e-9


# ---- Model circuit ----

def test_model_circuit_shape():
    rng = np.random.default_rng(0)
    mc = model_circuit(d=3, rng=rng)
    assert mc["final_state"].shape == (8,)
    assert mc["n_qubits"] == 3
    assert mc["depth"] == 3


def test_model_circuit_state_normalized():
    rng = np.random.default_rng(0)
    for d in (2, 3, 4):
        mc = model_circuit(d, rng)
        assert abs(np.linalg.norm(mc["final_state"]) - 1.0) < 1e-9


def test_model_circuit_n_layers():
    rng = np.random.default_rng(0)
    mc = model_circuit(d=4, rng=rng)
    assert len(mc["permutations"]) == 4
    assert len(mc["gates"]) == 4


def test_model_circuit_gates_per_layer():
    """Each layer has d/2 (rounded down) gates."""
    rng = np.random.default_rng(0)
    for d in (2, 3, 4, 5):
        mc = model_circuit(d, rng)
        for layer_gates in mc["gates"]:
            assert len(layer_gates) == d // 2


# ---- Heavy outputs ----

def test_heavy_output_set_uniform():
    probs = np.ones(4) / 4
    # All entries equal to median → no entries STRICTLY above median.
    heavy = heavy_output_set(probs)
    assert len(heavy) == 0


def test_heavy_output_set_skewed():
    probs = np.array([0.4, 0.3, 0.2, 0.1])
    heavy = heavy_output_set(probs)
    # sorted probs: [0.1, 0.2, 0.3, 0.4]. Median = sorted[2] = 0.3.
    # Strictly above 0.3: only index 0 (p=0.4).
    assert heavy == {0}


def test_heavy_output_probability_in_range():
    rng = np.random.default_rng(0)
    mc = model_circuit(d=4, rng=rng)
    p = np.abs(mc["final_state"]) ** 2
    h = heavy_output_probability(p)
    assert 0.0 <= h <= 1.0


def test_heavy_output_probability_approaches_porter_thomas():
    """Average heavy probability for ideal random circuits → (1+ln 2)/2 ≈ 0.847."""
    rng = np.random.default_rng(0)
    avgs = []
    for _ in range(30):
        mc = model_circuit(d=4, rng=rng)
        p = np.abs(mc["final_state"]) ** 2
        avgs.append(heavy_output_probability(p))
    mean = float(np.mean(avgs))
    # For d=4 (small dim), the Porter-Thomas asymptote is approached.
    assert 0.6 < mean < 0.95


# ---- Depolarizing noise ----

def test_depolarizing_zero_is_identity():
    probs = np.array([0.6, 0.2, 0.15, 0.05])
    out = _apply_depolarizing(probs, 0.0)
    assert np.allclose(out, probs)


def test_depolarizing_one_is_uniform():
    probs = np.array([0.6, 0.2, 0.15, 0.05])
    out = _apply_depolarizing(probs, 1.0)
    assert np.allclose(out, [0.25, 0.25, 0.25, 0.25])


# ---- QV trial / estimate ----

def test_qv_trial_returns_valid_structure():
    rng = np.random.default_rng(0)
    t = quantum_volume_trial(d=3, noise_level=0.0, rng=rng)
    assert "d" in t and "was_heavy" in t and "sampled_bitstring" in t
    assert t["d"] == 3


def test_qv_estimate_no_noise_passes_d3():
    """Without noise, d=3 should pass the 2/3 threshold easily."""
    rng = np.random.default_rng(0)
    r = quantum_volume_estimate(d=3, n_trials=100, noise_level=0.0, rng=rng)
    assert r["passes_2/3_threshold"]
    assert r["heavy_rate"] > 0.65


def test_qv_estimate_full_noise_fails():
    """At ~100% depolarizing noise, heavy rate → 0.5 (random)."""
    rng = np.random.default_rng(0)
    r = quantum_volume_estimate(d=3, n_trials=100, noise_level=1.0, rng=rng)
    assert not r["passes_2/3_threshold"]


# ---- find_quantum_volume ----

def test_find_qv_no_noise_returns_max_d():
    """Without noise, find_quantum_volume should hit max_d."""
    rng = np.random.default_rng(0)
    r = find_quantum_volume(max_d=4, n_trials_per_d=80,
                             noise_level=0.0, rng=rng)
    assert r["largest_d"] >= 3
    assert r["quantum_volume"] == 2 ** r["largest_d"]


def test_find_qv_high_noise_low_qv():
    """High noise → small QV (often 1)."""
    rng = np.random.default_rng(0)
    r = find_quantum_volume(max_d=4, n_trials_per_d=80,
                             noise_level=0.95, rng=rng)
    assert r["quantum_volume"] <= 8     # certainly should not exceed d=3


def test_find_qv_returns_per_d_results():
    rng = np.random.default_rng(0)
    r = find_quantum_volume(max_d=3, n_trials_per_d=20, rng=rng)
    assert "per_d_results" in r
    assert len(r["per_d_results"]) == 2     # d=2 and d=3
