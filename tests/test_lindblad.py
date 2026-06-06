"""Lindblad master equation tests."""

import numpy as np
import pytest

from qbit_simulator.lindblad import (
    lindblad_superoperator, evolve_density_matrix,
    _effective_hamiltonian,
    quantum_trajectory_step, simulate_trajectory, simulate_ensemble,
    amplitude_damping_jump_single_qubit,
    dephasing_jump_single_qubit,
    embed_single_qubit_op,
)


# ---- Superoperator construction ----

def test_superoperator_shape():
    H = np.eye(2, dtype=complex)
    L = lindblad_superoperator(H, [], [])
    assert L.shape == (4, 4)


def test_superoperator_n2():
    H = np.eye(4, dtype=complex)
    L = lindblad_superoperator(H, [], [])
    assert L.shape == (16, 16)


def test_superoperator_zero_with_zero_inputs():
    H = np.zeros((2, 2), dtype=complex)
    L = lindblad_superoperator(H, [], [])
    assert np.allclose(L, np.zeros((4, 4)), atol=1e-12)


def test_superoperator_unitary_part_traceless():
    """The unitary-only Lindbladian (no jump ops) is trace-preserving on
    density matrices: L · vec(I) = 0."""
    H = np.array([[0.5, 0.3], [0.3, -0.2]], dtype=complex)
    L = lindblad_superoperator(H, [], [])
    vec_I = np.eye(2, dtype=complex).flatten(order="F")
    assert np.allclose(L @ vec_I, np.zeros(4), atol=1e-12)


# ---- Exact evolution ----

def test_evolve_zero_time_is_identity():
    rho0 = np.array([[0.6, 0.1], [0.1, 0.4]], dtype=complex)
    L = lindblad_superoperator(np.zeros((2, 2), dtype=complex), [], [])
    rho_t = evolve_density_matrix(rho0, L, t=0.0)
    assert np.allclose(rho_t, rho0, atol=1e-12)


def test_evolve_amplitude_damping():
    """For pure damping, ρ_11(t) = ρ_11(0) · e^{-γ t}."""
    H = np.zeros((2, 2), dtype=complex)
    jump = amplitude_damping_jump_single_qubit()
    gamma = 0.7
    L = lindblad_superoperator(H, [jump], [gamma])
    rho0 = np.array([[0, 0], [0, 1]], dtype=complex)
    t = 1.5
    rho_t = evolve_density_matrix(rho0, L, t)
    expected = np.exp(-gamma * t)
    assert abs(rho_t[1, 1].real - expected) < 1e-10


def test_evolve_preserves_trace():
    H = np.array([[0.5, 0.3], [0.3, -0.2]], dtype=complex)
    L = lindblad_superoperator(
        H, [dephasing_jump_single_qubit()], [0.5],
    )
    rho0 = np.array([[0.6, 0.4], [0.4, 0.4]], dtype=complex)
    rho_t = evolve_density_matrix(rho0, L, t=1.0)
    assert abs(np.trace(rho_t).real - 1.0) < 1e-10


def test_evolve_preserves_hermiticity():
    H = np.array([[0.5, 0.3], [0.3, -0.2]], dtype=complex)
    L = lindblad_superoperator(H, [dephasing_jump_single_qubit()], [0.3])
    rho0 = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
    rho_t = evolve_density_matrix(rho0, L, t=2.0)
    assert np.allclose(rho_t, rho_t.conj().T, atol=1e-10)


def test_dephasing_kills_offdiagonals():
    """Pure dephasing should decay off-diagonal entries of the density
    matrix to zero, preserving diagonals."""
    H = np.zeros((2, 2), dtype=complex)
    L = lindblad_superoperator(H, [dephasing_jump_single_qubit()], [1.0])
    rho0 = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
    rho_t = evolve_density_matrix(rho0, L, t=10.0)
    assert abs(rho_t[0, 1]) < 1e-3
    assert abs(rho_t[0, 0].real - 0.5) < 1e-9


# ---- Effective Hamiltonian ----

def test_effective_hamiltonian_no_jumps_is_identity():
    H = np.array([[1, 0.5], [0.5, -1]], dtype=complex)
    H_eff = _effective_hamiltonian(H, [], [])
    assert np.allclose(H_eff, H, atol=1e-12)


def test_effective_hamiltonian_anti_hermitian_part():
    """H_eff = H - (i/2) γ L†L; for L = σ_-, L†L = |1⟩⟨1|."""
    H = np.zeros((2, 2), dtype=complex)
    L_op = amplitude_damping_jump_single_qubit()
    gamma = 0.5
    H_eff = _effective_hamiltonian(H, [L_op], [gamma])
    expected = -0.5j * gamma * np.array([[0, 0], [0, 1]], dtype=complex)
    assert np.allclose(H_eff, expected, atol=1e-12)


# ---- Trajectories ----

def test_trajectory_step_normalizes():
    H = np.zeros((2, 2), dtype=complex)
    rng = np.random.default_rng(0)
    psi = np.array([0, 1], dtype=complex)
    psi_new = quantum_trajectory_step(
        psi, H, [amplitude_damping_jump_single_qubit()], [0.1],
        dt=0.1, rng=rng,
    )
    assert abs(np.linalg.norm(psi_new) - 1.0) < 1e-10


def test_simulate_trajectory_returns_state_vector():
    H = np.zeros((2, 2), dtype=complex)
    rng = np.random.default_rng(0)
    psi0 = np.array([0, 1], dtype=complex)
    psi_T = simulate_trajectory(
        psi0, H, [amplitude_damping_jump_single_qubit()], [0.3],
        t=1.0, n_steps=20, rng=rng,
    )
    assert psi_T.shape == (2,)
    assert abs(np.linalg.norm(psi_T) - 1.0) < 1e-10


def test_ensemble_average_matches_exact():
    """MC ensemble average should approximate the exact ρ(t)."""
    H = np.zeros((2, 2), dtype=complex)
    L = amplitude_damping_jump_single_qubit()
    gamma = 0.5
    L_super = lindblad_superoperator(H, [L], [gamma])
    rho0 = np.array([[0, 0], [0, 1]], dtype=complex)
    rho_exact = evolve_density_matrix(rho0, L_super, t=1.0)
    rng = np.random.default_rng(0)
    psi0 = np.array([0, 1], dtype=complex)
    r = simulate_ensemble(psi0, H, [L], [gamma],
                            t=1.0, n_steps=100, n_traj=500, rng=rng)
    assert abs(r["rho"][1, 1].real - rho_exact[1, 1].real) < 0.05


def test_ensemble_returns_valid_density_matrix():
    """Output should be Hermitian, PSD, trace 1."""
    H = np.zeros((2, 2), dtype=complex)
    L = amplitude_damping_jump_single_qubit()
    rng = np.random.default_rng(0)
    psi0 = np.array([0, 1], dtype=complex)
    r = simulate_ensemble(psi0, H, [L], [0.3], t=0.5,
                           n_steps=20, n_traj=50, rng=rng)
    rho = r["rho"]
    assert np.allclose(rho, rho.conj().T, atol=1e-10)
    assert abs(np.trace(rho).real - 1.0) < 1e-10
    eigs = np.linalg.eigvalsh(rho)
    assert eigs.min() >= -1e-10


# ---- Embedding ----

def test_embed_single_qubit_op_correct_position():
    """Embed σ_- on qubit 1 of 3-qubit system."""
    sigma_minus = amplitude_damping_jump_single_qubit()
    op = embed_single_qubit_op(sigma_minus, qubit=1, n_qubits=3)
    assert op.shape == (8, 8)
    # On |010⟩ (idx 2), σ_- on qubit 1 → |000⟩ (idx 0).
    psi = np.zeros(8, dtype=complex); psi[2] = 1.0
    out = op @ psi
    assert abs(out[0] - 1.0) < 1e-12


def test_embed_qubit_zero():
    sigma_minus = amplitude_damping_jump_single_qubit()
    op = embed_single_qubit_op(sigma_minus, qubit=0, n_qubits=2)
    psi = np.zeros(4, dtype=complex); psi[2] = 1.0    # |10⟩
    out = op @ psi
    # σ_- on qubit 0 of |10⟩ → |00⟩ (idx 0).
    assert abs(out[0] - 1.0) < 1e-12
