"""Tests for parameter-shift gradients, Adam, and quantum natural gradient."""

import numpy as np
import pytest

from qbit_simulator.optimizers import (
    parameter_shift_gradient, adam, quantum_natural_gradient,
)


# ---- parameter-shift gradient ----

def test_parameter_shift_on_simple_cost():
    """For cost(θ) = sin(θ), ∂C/∂θ = cos(θ). Parameter-shift with s=π/2
    gives (sin(θ+π/2) - sin(θ-π/2))/(2 sin(π/2)) = (cos(θ)+cos(θ))/2 = cos(θ).
    """
    def cost(theta):
        return float(np.sin(theta[0]))
    for theta in (0.1, 0.5, 1.2, 2.5):
        g = parameter_shift_gradient(cost, np.array([theta]))
        assert abs(g[0] - np.cos(theta)) < 1e-9


def test_parameter_shift_on_vector_param():
    """Multi-parameter cost."""
    def cost(theta):
        return float(np.sin(theta[0]) + 2 * np.sin(theta[1]))
    theta = np.array([0.4, 0.8])
    g = parameter_shift_gradient(cost, theta)
    assert abs(g[0] - np.cos(theta[0])) < 1e-9
    assert abs(g[1] - 2 * np.cos(theta[1])) < 1e-9


# ---- Adam ----

def test_adam_minimizes_quadratic():
    """For cost(θ) = (θ - 1.5)², minimum is θ = 1.5."""
    def cost(theta):
        return float((theta[0] - 1.5) ** 2)
    result = adam(cost, np.array([0.0]),
                   lr=0.1, n_iter=300)
    assert abs(result["theta_opt"][0] - 1.5) < 0.05


def test_adam_finds_h2_ground_state_via_param_shift():
    """Apply Adam to H2 STO-3G VQE."""
    from qbit_simulator.algorithms.h2_sto3g import h2_sto3g_hamiltonian
    from qbit_simulator.algorithms.vqe import h2_ansatz

    H = h2_sto3g_hamiltonian(0.74)
    H_matrix = H.matrix()

    def cost(theta_vec):
        qc = h2_ansatz(float(theta_vec[0]))
        psi = qc.state
        return float(np.real(psi.conj() @ H_matrix @ psi))

    result = adam(cost, np.array([0.3]), lr=0.05, n_iter=200)
    e_exact, _ = H.ground_state()
    assert abs(result["cost_opt"] - e_exact) < 1e-3


def test_adam_history_records_each_iteration():
    def cost(theta):
        return float((theta[0]) ** 2)
    result = adam(cost, np.array([1.0]), lr=0.1, n_iter=50)
    # History should have at least a few entries (may terminate early).
    assert len(result["history"]) >= 5


# ---- Quantum natural gradient ----

def test_qng_converges_on_simple_quantum_cost():
    """For a single-parameter ansatz Ry(θ)|0⟩ with cost = -⟨Z⟩, minimum is
    θ = π. Test that QNG converges."""
    from qbit_simulator.gates import Z
    Z_op = Z

    def state_fn(theta):
        # Ry(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
        return np.array([np.cos(theta[0] / 2), np.sin(theta[0] / 2)],
                         dtype=np.complex128)

    def cost(theta):
        psi = state_fn(theta)
        return float(np.real(psi.conj() @ Z_op @ psi))

    result = quantum_natural_gradient(cost, state_fn, np.array([0.5]),
                                       lr=0.1, n_iter=80)
    # Minimum is at θ = π where ⟨Z⟩ = -1.
    assert result["cost_opt"] < -0.9
