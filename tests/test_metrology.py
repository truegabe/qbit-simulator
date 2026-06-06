"""Quantum metrology tests — SQL vs Heisenberg-limited phase estimation."""

import numpy as np
import pytest

from qbit_simulator.algorithms.metrology import (
    sql_phase_estimate, heisenberg_phase_estimate, metrology_comparison,
    build_ghz_state,
)


# ---- SQL ----

def test_sql_estimate_accuracy():
    """For φ = 0.1 with many shots, SQL estimator should be close to 0.1."""
    rng = np.random.default_rng(0)
    r = sql_phase_estimate(phi=0.1, n_qubits=10, n_shots=5000, rng=rng)
    assert abs(r["phi_estimate"] - 0.1) < 0.05


def test_sql_uncertainty_scales_with_inverse_sqrt_N():
    """SQL: σ_φ ~ 1/√N. Doubling N → uncertainty drops by √2."""
    rng = np.random.default_rng(0)
    r_small = sql_phase_estimate(phi=0.3, n_qubits=10, n_shots=2000,
                                  rng=np.random.default_rng(0))
    r_large = sql_phase_estimate(phi=0.3, n_qubits=40, n_shots=2000,
                                  rng=np.random.default_rng(0))
    # uncertainty(N=40) / uncertainty(N=10) should be ~ 1/√4 = 0.5.
    ratio = r_large["uncertainty"] / r_small["uncertainty"]
    assert 0.35 < ratio < 0.7


# ---- Heisenberg-limited ----

def test_heisenberg_estimate_accuracy():
    """For small φ, the Heisenberg estimator should recover φ accurately."""
    rng = np.random.default_rng(0)
    r = heisenberg_phase_estimate(phi=0.1, n_qubits=4, n_shots=5000, rng=rng)
    assert abs(r["phi_estimate"] - 0.1) < 0.02


def test_heisenberg_uncertainty_scales_inverse_N():
    """Heisenberg: σ_φ ~ 1/N. Doubling N → uncertainty drops by 2x."""
    rng = np.random.default_rng(0)
    r_small = heisenberg_phase_estimate(phi=0.05, n_qubits=4, n_shots=2000,
                                          rng=np.random.default_rng(0))
    r_large = heisenberg_phase_estimate(phi=0.05, n_qubits=8, n_shots=2000,
                                          rng=np.random.default_rng(0))
    ratio = r_large["uncertainty"] / r_small["uncertainty"]
    assert 0.3 < ratio < 0.8   # √(N_large/N_small) factor


# ---- Comparison ----

def test_heisenberg_beats_sql():
    """With equal shot count, HL should give a smaller uncertainty than SQL."""
    rng = np.random.default_rng(0)
    cmp = metrology_comparison(phi=0.1, n_qubits=8, n_shots=8000, rng=rng)
    # HL uncertainty < SQL uncertainty.
    assert cmp["heisenberg"]["uncertainty"] < cmp["sql"]["uncertainty"]
    # Speedup should be roughly √N = √8 ≈ 2.83 (with sampling noise).
    assert cmp["speedup_ratio"] > 1.5


# ---- GHZ resource ----

def test_ghz_state_is_normalized():
    psi = build_ghz_state(5)
    assert abs(np.linalg.norm(psi) - 1.0) < 1e-10


def test_ghz_state_has_two_components():
    psi = build_ghz_state(4)
    # Only |0000⟩ and |1111⟩ should be nonzero.
    nonzero_indices = [i for i, a in enumerate(psi) if abs(a) > 1e-9]
    assert set(nonzero_indices) == {0, 15}
    for i in nonzero_indices:
        assert abs(psi[i] - 1 / np.sqrt(2)) < 1e-9
