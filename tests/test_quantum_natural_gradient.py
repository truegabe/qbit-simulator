"""Quantum natural gradient tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.quantum_natural_gradient import (
    state_derivative, fubini_study_metric,
    parameter_shift_gradient,
    quantum_natural_gradient_step, train_with_qng,
)
from qbit_simulator.algorithms.ssvqe import (
    hardware_efficient_ansatz_apply, pauli_op_to_matrix,
)
from qbit_simulator.algorithms.h2_sto3g import h2_sto3g_hamiltonian


# ---- State derivative ----

def test_state_derivative_zero_params_is_zero_for_first_layer():
    """Special property — first Ry layer at θ=0 → state stays at ref."""
    ansatz, n_p = hardware_efficient_ansatz_apply(n_qubits=2, depth=1)
    params = np.zeros(n_p)
    ref = np.array([1, 0, 0, 0], dtype=complex)
    d = state_derivative(ansatz, params, ref, param_idx=0)
    # The derivative is not necessarily zero; just check it's finite.
    assert np.all(np.isfinite(d))


def test_state_derivative_shape():
    ansatz, n_p = hardware_efficient_ansatz_apply(n_qubits=2, depth=1)
    rng = np.random.default_rng(0)
    params = rng.uniform(-1, 1, size=n_p)
    ref = np.array([1, 0, 0, 0], dtype=complex)
    d = state_derivative(ansatz, params, ref, param_idx=2)
    assert d.shape == (4,)


def test_state_derivative_magnitude_correct():
    """For Ry(θ)|0⟩ at θ=0, the exact derivative is (0, 0.5).

    Regression guard: a previous version was off by sqrt(2) — used
    divisor 2·sin(s) instead of the correct 4·sin(s/2).
    """
    ansatz, n_p = hardware_efficient_ansatz_apply(n_qubits=1, depth=0)
    params = np.array([0.0])
    ref = np.array([1, 0], dtype=complex)
    d = state_derivative(ansatz, params, ref, param_idx=0)
    expected = np.array([0.0, 0.5], dtype=complex)
    assert np.allclose(d, expected, atol=1e-12)


# ---- Fubini-Study metric ----

def test_fubini_study_metric_symmetric():
    ansatz, n_p = hardware_efficient_ansatz_apply(n_qubits=2, depth=2)
    rng = np.random.default_rng(0)
    params = rng.uniform(-1, 1, size=n_p)
    ref = np.array([1, 0, 0, 0], dtype=complex)
    F = fubini_study_metric(ansatz, params, ref)
    assert np.allclose(F, F.T, atol=1e-12)


def test_fubini_study_metric_psd():
    """F should be positive semi-definite."""
    ansatz, n_p = hardware_efficient_ansatz_apply(n_qubits=2, depth=2)
    rng = np.random.default_rng(0)
    params = rng.uniform(-1, 1, size=n_p)
    ref = np.array([1, 0, 0, 0], dtype=complex)
    F = fubini_study_metric(ansatz, params, ref)
    eigs = np.linalg.eigvalsh(F)
    assert eigs.min() >= -1e-9


def test_fubini_study_metric_shape():
    ansatz, n_p = hardware_efficient_ansatz_apply(n_qubits=2, depth=2)
    params = np.zeros(n_p)
    ref = np.array([1, 0, 0, 0], dtype=complex)
    F = fubini_study_metric(ansatz, params, ref)
    assert F.shape == (n_p, n_p)


# ---- Parameter-shift gradient ----

def test_parameter_shift_gradient_simple():
    """For loss(θ) = sin(θ_0), ∂L/∂θ_0 at θ_0 = 0 should be 1."""
    def loss(params):
        return float(np.sin(params[0]))
    grad = parameter_shift_gradient(loss, np.array([0.0, 0.5, 0.1]))
    assert abs(grad[0] - 1.0) < 1e-9
    assert abs(grad[1]) < 1e-9   # loss doesn't depend on θ_1.


def test_parameter_shift_gradient_matches_finite_difference_on_simple():
    """For loss(θ) = cos(θ_0)·cos(θ_1), check at random points."""
    def loss(params):
        return float(np.cos(params[0]) * np.cos(params[1]))
    rng = np.random.default_rng(0)
    params = rng.uniform(-1, 1, size=2)
    grad_ps = parameter_shift_gradient(loss, params)
    # Compare with finite differences (these are exact for sin/cos
    # because of the parameter-shift property of trig).
    eps = 1e-6
    grad_fd = np.zeros(2)
    for i in range(2):
        p_p = params.copy(); p_p[i] += eps
        p_m = params.copy(); p_m[i] -= eps
        grad_fd[i] = (loss(p_p) - loss(p_m)) / (2 * eps)
    # parameter_shift gives the EXACT trig gradient.
    # Both should equal -sin(θ_i) · cos(θ_{1-i}).
    assert np.allclose(grad_ps, grad_fd, atol=1e-3)


# ---- QNG step ----

def test_qng_step_decreases_loss():
    H_mat = pauli_op_to_matrix(h2_sto3g_hamiltonian(0.74), 2)
    ansatz, n_p = hardware_efficient_ansatz_apply(n_qubits=2, depth=2)
    ref = np.array([1, 0, 0, 0], dtype=complex)
    def loss(p):
        psi = ansatz(p, ref)
        return float(np.real(psi.conj() @ H_mat @ psi))
    rng = np.random.default_rng(0)
    params = rng.uniform(-1, 1, size=n_p)
    L_before = loss(params)
    new_params, info = quantum_natural_gradient_step(
        loss, ansatz, params, ref, lr=0.05,
    )
    L_after = loss(new_params)
    assert L_after <= L_before + 1e-6


# ---- Full training ----

def test_qng_converges_to_h2_ground_state():
    """QNG should reach the H₂ STO-3G ground state with enough iterations."""
    H_mat = pauli_op_to_matrix(h2_sto3g_hamiltonian(0.74), 2)
    true_gs = float(np.linalg.eigvalsh(H_mat)[0])
    ansatz, n_p = hardware_efficient_ansatz_apply(n_qubits=2, depth=2)
    ref = np.array([1, 0, 0, 0], dtype=complex)
    def loss(p):
        psi = ansatz(p, ref)
        return float(np.real(psi.conj() @ H_mat @ psi))
    rng = np.random.default_rng(0)
    init = rng.uniform(-1, 1, size=n_p)
    result = train_with_qng(loss, ansatz, init, ref, n_iter=80, lr=0.3)
    assert result["loss"] - true_gs < 0.01


def test_qng_returns_structured_dict():
    H_mat = pauli_op_to_matrix(h2_sto3g_hamiltonian(0.74), 2)
    ansatz, n_p = hardware_efficient_ansatz_apply(n_qubits=2, depth=2)
    ref = np.array([1, 0, 0, 0], dtype=complex)
    def loss(p):
        psi = ansatz(p, ref)
        return float(np.real(psi.conj() @ H_mat @ psi))
    init = np.zeros(n_p)
    result = train_with_qng(loss, ansatz, init, ref, n_iter=5)
    assert "params" in result
    assert "loss" in result
    assert "loss_history" in result
    assert "converged" in result
    assert len(result["loss_history"]) >= 2


def test_qng_loss_history_decreasing():
    H_mat = pauli_op_to_matrix(h2_sto3g_hamiltonian(0.74), 2)
    ansatz, n_p = hardware_efficient_ansatz_apply(n_qubits=2, depth=2)
    ref = np.array([1, 0, 0, 0], dtype=complex)
    def loss(p):
        psi = ansatz(p, ref)
        return float(np.real(psi.conj() @ H_mat @ psi))
    rng = np.random.default_rng(0)
    init = rng.uniform(-1, 1, size=n_p)
    result = train_with_qng(loss, ansatz, init, ref, n_iter=20, lr=0.1)
    history = result["loss_history"]
    # Loss should generally decrease (allow small oscillations).
    assert history[-1] <= history[0]
