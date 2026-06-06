"""Tests for Tier beta primitives:
    LCU, density-matrix exponentiation, generalized AA, Jordan gradient.
"""

import numpy as np
import pytest

from qbit_simulator.algorithms.lcu import (
    apply_lcu, lcu_unitary, prep_state, taylor_hamiltonian_simulation,
)
from qbit_simulator.algorithms.density_matrix_exponentiation import (
    dme_step, dme_evolve, swap_operator,
)
from qbit_simulator.algorithms.generalized_amplitude_amplification import (
    amplitude_amplification, amp_amp_with_oracle, optimal_iterations,
    success_probability, oblivious_amplitude_amplification,
)
from qbit_simulator.algorithms.quantum_gradient import (
    quantum_gradient, phase_oracle_diagonal,
)


# Single-qubit Paulis.
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


# ---- Tier beta.1: LCU ----

def test_prep_state_amplitudes_proportional_to_sqrt_alpha():
    alphas = np.array([0.5, 0.3, 0.2])
    state = prep_state(alphas)
    # First three amplitudes should be sqrt(alpha_k / sum).
    s = alphas.sum()
    expected = np.sqrt(alphas / s)
    assert np.allclose(state[:3].real, expected)
    # Padding entries are zero.
    assert state[3] == 0


def test_prep_state_normalized():
    alphas = np.array([1.0, 2.0, 1.5, 0.5])
    state = prep_state(alphas)
    assert abs(np.linalg.norm(state) - 1.0) < 1e-9


def test_lcu_implements_linear_combination():
    """A psi after LCU = sum_k alpha_k U_k psi, exactly."""
    psi = np.array([1.0, 0.0], dtype=complex)
    alphas = [0.6, 0.4]
    res = apply_lcu(alphas, [X, Z], psi)
    expected = 0.6 * (X @ psi) + 0.4 * (Z @ psi)
    norm = np.linalg.norm(expected)
    expected_norm = expected / norm
    assert np.allclose(res["psi_norm"], expected_norm, atol=1e-9)


def test_lcu_success_probability():
    """Success prob = ||A psi||^2 / (sum alpha)^2."""
    psi = np.array([1.0, 0.0], dtype=complex)
    alphas = [0.5, 0.5]
    res = apply_lcu(alphas, [X, X], psi)
    # (0.5 X + 0.5 X) = X. ||X|0>||^2 = 1. s=1.
    assert abs(res["prob"] - 1.0) < 1e-9


def test_lcu_unitary_is_unitary():
    W, n_anc, n_sys = lcu_unitary([1.0, 0.5, 0.3], [X, Z, I2])
    d = W.shape[0]
    assert np.allclose(W @ W.conj().T, np.eye(d), atol=1e-9)


def test_taylor_hamiltonian_simulation_matches_exact():
    psi = np.array([1.0, 0.0], dtype=complex)
    H_terms = [(0.5, X), (0.3, Z)]
    out = taylor_hamiltonian_simulation(H_terms, t=0.4, psi=psi, K=6)
    assert out["fidelity"] > 0.999


def test_taylor_simulation_higher_K_more_accurate():
    psi = np.array([1.0, 0.0], dtype=complex)
    H_terms = [(0.5, X), (0.5, Y)]
    out_low = taylor_hamiltonian_simulation(H_terms, t=0.5, psi=psi, K=2)
    out_high = taylor_hamiltonian_simulation(H_terms, t=0.5, psi=psi, K=8)
    assert out_high["fidelity"] >= out_low["fidelity"]


# ---- Tier beta.2: Density-matrix exponentiation ----

def test_swap_operator_swaps_basis_states():
    S = swap_operator(2)
    # |01> -> |10>: index 0*2+1=1 -> index 1*2+0=2.
    psi = np.zeros(4); psi[1] = 1.0
    assert np.allclose(S @ psi, np.array([0, 0, 1, 0]))


def test_swap_operator_unitary():
    S = swap_operator(2)
    assert np.allclose(S @ S, np.eye(4))


def test_dme_step_preserves_trace_and_hermiticity():
    rng = np.random.default_rng(0)
    d = 2
    A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    rho = A @ A.conj().T; rho /= np.trace(rho)
    sigma = np.eye(d, dtype=complex) / d
    sigma_new = dme_step(sigma, rho, dt=0.1)
    assert abs(np.real(np.trace(sigma_new)) - 1.0) < 1e-9
    assert np.allclose(sigma_new, sigma_new.conj().T, atol=1e-9)


def test_dme_converges_to_exact():
    rng = np.random.default_rng(1)
    d = 2
    A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    rho = A @ A.conj().T; rho /= np.trace(rho)
    sigma0 = np.array([[0.7, 0.1], [0.1, 0.3]], dtype=complex)
    sigma0 = 0.5 * (sigma0 + sigma0.conj().T); sigma0 /= np.trace(sigma0)
    out_coarse = dme_evolve(sigma0, rho, t=0.5, n_steps=20)
    out_fine   = dme_evolve(sigma0, rho, t=0.5, n_steps=500)
    err_coarse = np.linalg.norm(out_coarse["sigma"] - out_coarse["sigma_exact"])
    err_fine   = np.linalg.norm(out_fine["sigma"] - out_fine["sigma_exact"])
    assert err_fine < err_coarse


def test_dme_identity_when_rho_commutes():
    """If rho commutes with sigma, DME should keep sigma unchanged."""
    d = 2
    rho = np.diag([0.7, 0.3]).astype(complex)
    sigma = np.diag([0.5, 0.5]).astype(complex)
    out = dme_evolve(sigma, rho, t=0.3, n_steps=50)
    # sigma_exact = exp(-i rho t) sigma exp(+i rho t) = sigma for diagonal.
    assert np.allclose(out["sigma"], sigma, atol=1e-3)


# ---- Tier beta.3: Generalized amplitude amplification ----

def test_amp_amp_grover_3qubit():
    n = 3; d = 2 ** n
    H1 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    H = H1
    for _ in range(n - 1):
        H = np.kron(H, H1)
    out = amp_amp_with_oracle(H, marked=[5])
    assert out["prob_final"] > 0.9


def test_optimal_iterations_matches_grover():
    # For N=8 (3-qubit Grover), k* = round((pi/4)*sqrt(N) - 1/2) ≈ 2.
    assert optimal_iterations(1 / 8) == 2
    # For N=16, k* ≈ 3.
    assert optimal_iterations(1 / 16) == 3


def test_success_probability_formula():
    # sin^2((2k+1)theta) where sin^2(theta) = 1/4.
    p = success_probability(0.25, 1)
    # theta = pi/6. (2*1+1)*pi/6 = pi/2. sin^2(pi/2) = 1.
    assert abs(p - 1.0) < 1e-9


def test_amp_amp_arbitrary_initial_state():
    rng = np.random.default_rng(0)
    d = 8
    A = np.linalg.qr(rng.standard_normal((d, d))
                       + 1j * rng.standard_normal((d, d)))[0]
    out = amp_amp_with_oracle(A, marked=[2])
    # Final prob >= initial prob (amplification at least preserves).
    assert out["prob_final"] >= out["prob_initial"]


def test_oblivious_amp_amp_returns_unitary():
    n_anc = 1; n_sys = 1
    d = 4
    rng = np.random.default_rng(0)
    W = np.linalg.qr(rng.standard_normal((d, d))
                       + 1j * rng.standard_normal((d, d)))[0]
    out = oblivious_amplitude_amplification(W, n_anc=n_anc, n_iters=1)
    U = out["U_oaa"]
    assert np.allclose(U @ U.conj().T, np.eye(d), atol=1e-9)


# ---- Tier beta.4: Quantum gradient ----

def test_quantum_gradient_1d_linear():
    out = quantum_gradient(lambda x: 2.5 * x[0],
                              x0=np.array([0.3]),
                              bits=6, M=4.0, step=0.5)
    assert abs(out["gradient"][0] - 2.5) < 1e-6


def test_quantum_gradient_2d_linear():
    out = quantum_gradient(lambda x: 3.0 * x[0] + 2.0 * x[1],
                              x0=np.array([0.2, 0.5]),
                              bits=6, M=4.0, step=0.5)
    assert np.allclose(out["gradient"], [3.0, 2.0], atol=1e-6)


def test_quantum_gradient_classical_reference():
    out = quantum_gradient(lambda x: np.sin(x[0]),
                              x0=np.array([0.3]),
                              bits=6, M=16.0, step=0.1)
    # Classical reference cos(0.3) ≈ 0.9553.
    assert abs(out["classical_gradient"][0] - np.cos(0.3)) < 1e-3


def test_quantum_gradient_nonlinear_within_quantization():
    # f = x^2 + 2y at (0.4, 0.3) -> grad = [0.8, 2.0].
    # With M=24, step=0.1: precision = 1/(M*step) ≈ 0.04.
    out = quantum_gradient(lambda x: x[0] ** 2 + 2 * x[1],
                              x0=np.array([0.4, 0.3]),
                              bits=6, M=24.0, step=0.1)
    # Both coordinates should be within ~0.1 of true.
    assert abs(out["gradient"][0] - 0.8) < 0.1
    assert abs(out["gradient"][1] - 2.0) < 0.2


def test_phase_oracle_diagonal_shape_1d():
    diag = phase_oracle_diagonal(lambda x: x[0], x0=np.array([0.0]),
                                    d=1, bits=4, M=1.0, step=1.0)
    assert diag.shape == (16,)
    assert np.allclose(np.abs(diag), 1.0)
