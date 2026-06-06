"""Tests for magic state distillation (Bravyi-Kitaev)."""

import numpy as np
import pytest

from qbit_simulator.magic_state_distillation import (
    T_STATE, T_STATE_DM,
    noisy_t_state_dm, t_state_fidelity,
    distill_5to1_density_matrix, distill_15to1_density_matrix,
    distillation_resource_cost,
    simulate_distillation_error, distillation_error_curve,
)


# ---- T state representation ----

def test_t_state_normalized():
    assert abs(np.linalg.norm(T_STATE) - 1.0) < 1e-12


def test_t_state_dm_trace_one():
    assert abs(np.trace(T_STATE_DM).real - 1.0) < 1e-12


def test_t_state_dm_hermitian():
    assert np.allclose(T_STATE_DM, T_STATE_DM.conj().T)


def test_noisy_state_zero_eps_is_pure():
    rho = noisy_t_state_dm(0.0)
    assert abs(t_state_fidelity(rho) - 1.0) < 1e-12


def test_noisy_state_full_eps_is_maximally_mixed():
    rho = noisy_t_state_dm(1.0)
    assert np.allclose(rho, np.eye(2) / 2)
    # Fidelity with |T⟩ is 1/2 for the maximally mixed state.
    assert abs(t_state_fidelity(rho) - 0.5) < 1e-12


def test_noisy_state_linear_interpolation():
    """Fidelity should be 1 - eps/2 for depolarizing noise."""
    for eps in [0.01, 0.1, 0.3, 0.5]:
        rho = noisy_t_state_dm(eps)
        f = t_state_fidelity(rho)
        # ρ = (1-ε)|T⟩⟨T| + ε I/2 → ⟨T|ρ|T⟩ = (1-ε) + ε·(1/2) = 1 - ε/2.
        assert abs(f - (1 - eps / 2)) < 1e-12


def test_noisy_state_rejects_out_of_range():
    with pytest.raises(ValueError):
        noisy_t_state_dm(-0.1)
    with pytest.raises(ValueError):
        noisy_t_state_dm(1.5)


# ---- 5-to-1 analytic model ----

def test_5to1_zero_input_zero_output():
    p_success, eps_out = distill_5to1_density_matrix(0.0)
    assert abs(p_success - 1.0) < 1e-12
    assert eps_out == 0.0


def test_5to1_quadratic_suppression():
    """At small ε, eps_out should scale as ε²."""
    eps_small = 1e-3
    _, eps_out = distill_5to1_density_matrix(eps_small)
    # 5 · ε² · (1-ε)³ ≈ 5 · ε² at small ε. Diff is O(ε³) ~ 1.5e-8.
    assert abs(eps_out - 5 * eps_small ** 2) < 5e-8


def test_5to1_p_success_decreases_with_eps():
    ps = [distill_5to1_density_matrix(e)[0] for e in [0.0, 0.05, 0.1, 0.2]]
    for a, b in zip(ps, ps[1:]):
        assert a > b


# ---- 15-to-1 analytic model ----

def test_15to1_zero_input_zero_output():
    p_success, eps_out = distill_15to1_density_matrix(0.0)
    assert abs(p_success - 1.0) < 1e-12
    assert eps_out == 0.0


def test_15to1_cubic_suppression():
    """eps_out ≈ 35 · ε³."""
    eps_small = 1e-3
    _, eps_out = distill_15to1_density_matrix(eps_small)
    assert abs(eps_out - 35 * eps_small ** 3) < 1e-8


def test_15to1_beats_5to1_at_small_eps():
    """At small ε, cubic suppression dominates quadratic."""
    eps = 0.01
    _, e5 = distill_5to1_density_matrix(eps)
    _, e15 = distill_15to1_density_matrix(eps)
    assert e15 < e5


# ---- Resource cost ----

def test_resource_cost_no_layers_needed():
    """If raw ε already meets target, no distillation needed."""
    r = distillation_resource_cost(target_eps=0.5, raw_eps=0.01,
                                    protocol="15to1")
    assert r["layers"] == 0
    assert r["n_raw_states"] == 1


def test_resource_cost_one_layer_15to1():
    """If 35·ε³ < target < ε, exactly one layer suffices."""
    raw_eps = 0.01
    # 35·(0.01)³ = 3.5e-5
    target = 1e-4
    r = distillation_resource_cost(target, raw_eps, protocol="15to1")
    assert r["layers"] == 1
    assert r["n_raw_states"] == 15


def test_resource_cost_recursion():
    """Two-layer 15-to-1 should give 15² = 225 raw states."""
    raw_eps = 0.05
    target = 1e-10
    r = distillation_resource_cost(target, raw_eps, protocol="15to1")
    assert r["layers"] >= 2
    assert r["n_raw_states"] == 15 ** r["layers"]


def test_resource_cost_invalid_protocol():
    with pytest.raises(ValueError):
        distillation_resource_cost(0.01, 0.1, protocol="bogus")


# ---- Monte Carlo ----

def test_simulate_distillation_runs():
    rng = np.random.default_rng(0)
    r = simulate_distillation_error(0.05, protocol="15to1",
                                     n_trials=500, rng=rng)
    assert 0.0 <= r["p_success"] <= 1.0
    assert 0.0 <= r["eps_out"] <= 1.0
    assert r["protocol"] == "15to1"


def test_simulate_5to1_low_eps_high_success():
    rng = np.random.default_rng(0)
    r = simulate_distillation_error(0.01, protocol="5to1",
                                     n_trials=2000, rng=rng)
    # At low ε, almost all trials should post-select.
    assert r["p_success"] > 0.95


def test_simulate_15to1_matches_analytic_cubic():
    """The MC and the analytic model should agree on cubic suppression."""
    rng = np.random.default_rng(0)
    eps = 0.05
    r = simulate_distillation_error(eps, "15to1", n_trials=50000, rng=rng)
    _, eps_analytic = distill_15to1_density_matrix(eps)
    # MC has Poisson shot noise but should be within a factor of ~2-3.
    assert 0.3 * eps_analytic < r["eps_out"] < 3.0 * eps_analytic


def test_simulate_15to1_psuccess_matches_analytic():
    """P_success ≈ (1 - ε)^15 at the leading order."""
    rng = np.random.default_rng(0)
    eps = 0.03
    r = simulate_distillation_error(eps, "15to1", n_trials=20000, rng=rng)
    expected_p_success = (1 - eps) ** 15
    assert abs(r["p_success"] - expected_p_success) < 0.02


def test_distillation_error_curve_shape():
    rng = np.random.default_rng(0)
    eps_values = [0.01, 0.02, 0.05, 0.1]
    r = distillation_error_curve(eps_values, protocol="15to1",
                                  n_trials=500, rng=rng)
    assert len(r["eps_in"]) == 4
    assert len(r["eps_out"]) == 4
    assert len(r["p_success"]) == 4
