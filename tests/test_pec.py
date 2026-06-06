"""Probabilistic Error Cancellation tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.pec import (
    depolarizing_inverse_quasiprobs, pec_sampling_cost,
    apply_pauli_to_state, pec_estimator,
)


# ---- Quasi-probabilities ----

def test_quasiprobs_zero_noise_is_identity_only():
    """At p=0, η_I = 1 and all other η's = 0."""
    qp = depolarizing_inverse_quasiprobs(0.0, n_qubits=1)
    # eta_I = 1, all others = 0.
    assert abs(qp["eta"][0] - 1.0) < 1e-12
    assert all(abs(e) < 1e-12 for e in qp["eta"][1:])


def test_quasiprobs_count_for_n_qubits():
    """n_qubits → 4^n entries in the η-list."""
    qp1 = depolarizing_inverse_quasiprobs(0.1, n_qubits=1)
    qp2 = depolarizing_inverse_quasiprobs(0.1, n_qubits=2)
    assert len(qp1["eta"]) == 4
    assert len(qp2["eta"]) == 16


def test_quasiprobs_gamma_grows_with_noise():
    g_small = depolarizing_inverse_quasiprobs(0.01, 1)["gamma"]
    g_large = depolarizing_inverse_quasiprobs(0.2, 1)["gamma"]
    assert g_large > g_small


def test_quasiprobs_eta_I_positive():
    """The identity coefficient should remain positive at moderate noise."""
    qp = depolarizing_inverse_quasiprobs(0.1, 1)
    assert qp["eta"][0] > 0


def test_quasiprobs_other_eta_negative():
    """Non-identity Pauli coefficients should be negative for inverting
    a contractive channel."""
    qp = depolarizing_inverse_quasiprobs(0.1, 1)
    for e in qp["eta"][1:]:
        assert e < 0


def test_quasiprobs_operations_match_eta_length():
    qp = depolarizing_inverse_quasiprobs(0.1, 2)
    assert len(qp["operations"]) == len(qp["eta"])


# ---- Sampling cost ----

def test_pec_sampling_cost_zero_noise_is_one():
    eta = depolarizing_inverse_quasiprobs(0.0, 1)["eta"]
    assert abs(pec_sampling_cost(eta) - 1.0) < 1e-9


def test_pec_sampling_cost_increases_with_noise():
    c1 = pec_sampling_cost(depolarizing_inverse_quasiprobs(0.01, 1)["eta"])
    c2 = pec_sampling_cost(depolarizing_inverse_quasiprobs(0.1, 1)["eta"])
    assert c2 > c1


# ---- Apply Pauli ----

def test_apply_pauli_identity_is_noop():
    psi = np.array([0.6, 0.8], dtype=complex)
    out = apply_pauli_to_state(psi, "I")
    assert np.allclose(out, psi, atol=1e-12)


def test_apply_pauli_x_flips_basis():
    psi = np.array([1, 0], dtype=complex)
    out = apply_pauli_to_state(psi, "X")
    assert np.allclose(out, np.array([0, 1]), atol=1e-12)


def test_apply_pauli_preserves_norm():
    psi = np.array([0.6, 0.8], dtype=complex)
    for s in ("X", "Y", "Z"):
        out = apply_pauli_to_state(psi, s)
        assert abs(np.linalg.norm(out) - 1.0) < 1e-12


# ---- End-to-end PEC ----

def test_pec_reduces_bias():
    """PEC should bring the mitigated estimate closer to the ideal than
    the unmitigated one (on average)."""
    def ideal_state():
        return np.array([1, 1], dtype=complex) / np.sqrt(2)

    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    noise_rng = np.random.default_rng(42)
    p = 0.1

    def noisy(psi):
        r = noise_rng.uniform()
        if r < p / 3:
            return X @ psi
        elif r < 2 * p / 3:
            return Y @ psi
        elif r < p:
            return Z @ psi
        return psi

    rng = np.random.default_rng(0)
    r = pec_estimator(ideal_state, noisy, X, p_noise=p, n_shots=20000, rng=rng)
    bias_unmit = abs(r["unmitigated_estimate"] - r["ideal_value"])
    bias_mit = abs(r["mitigated_estimate"] - r["ideal_value"])
    # Mitigated should be closer to ideal.
    assert bias_mit < bias_unmit


def test_pec_returns_diagnostic_dict():
    def ideal_state():
        return np.array([1, 0], dtype=complex)

    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    rng = np.random.default_rng(0)
    r = pec_estimator(
        ideal_state, lambda psi: psi, Z, p_noise=0.05, n_shots=100, rng=rng,
    )
    for k in ("mitigated_estimate", "unmitigated_estimate",
              "ideal_value", "variance", "gamma", "sampling_overhead"):
        assert k in r


def test_pec_zero_noise_no_correction():
    """At p=0, mitigated and unmitigated should match the ideal."""
    def ideal_state():
        return np.array([1, 1], dtype=complex) / np.sqrt(2)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    rng = np.random.default_rng(0)
    r = pec_estimator(ideal_state, lambda psi: psi, X,
                       p_noise=0.0, n_shots=500, rng=rng)
    assert abs(r["mitigated_estimate"] - r["ideal_value"]) < 0.1
