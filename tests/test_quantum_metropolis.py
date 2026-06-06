"""Quantum metropolis / quantum simulated annealing tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.quantum_metropolis import (
    metropolis_sample, simulated_annealing,
    quantum_annealing_evolution, quantum_annealing_ground_state,
    transverse_field_initial_hamiltonian, _pauli_kron,
)


# ---- Classical Metropolis ----

def test_metropolis_returns_valid_states():
    rng = np.random.default_rng(0)
    r = metropolis_sample(lambda s: 0.0, n_bits=4, n_steps=100, beta=1.0, rng=rng)
    assert all(0 <= s < 16 for s in r["samples"])
    assert len(r["samples"]) == 101   # initial + n_steps


def test_metropolis_finds_low_energy_state():
    """Energy function = number of 1-bits (Hamming weight). Min is 0."""
    rng = np.random.default_rng(0)
    energy_fn = lambda s: float(bin(s).count("1"))
    r = metropolis_sample(energy_fn, n_bits=6, n_steps=5000, beta=5.0, rng=rng)
    assert r["min_energy"] == 0
    assert r["min_state"] == 0


def test_metropolis_acceptance_decreases_with_beta():
    """At high β, the chain rarely accepts uphill moves."""
    rng = np.random.default_rng(0)
    energy_fn = lambda s: float(bin(s).count("1"))
    r_cold = metropolis_sample(energy_fn, n_bits=4, n_steps=2000, beta=10.0, rng=rng)
    rng = np.random.default_rng(0)
    r_hot = metropolis_sample(energy_fn, n_bits=4, n_steps=2000, beta=0.1, rng=rng)
    assert r_cold["acceptance"] < r_hot["acceptance"]


def test_simulated_annealing_finds_optimum():
    """Annealing should reliably find the all-zeros minimum."""
    rng = np.random.default_rng(0)
    energy_fn = lambda s: float(bin(s).count("1"))
    r = simulated_annealing(energy_fn, n_bits=6, n_steps=5000, rng=rng)
    assert r["min_energy"] == 0


# ---- Quantum (adiabatic) ----

def test_transverse_field_hamiltonian_hermitian():
    H = transverse_field_initial_hamiltonian(3)
    assert np.allclose(H, H.conj().T, atol=1e-12)


def test_transverse_field_ground_state_is_uniform():
    """The GS of -sum X_i is |+...+⟩ = uniform superposition."""
    H = transverse_field_initial_hamiltonian(3)
    eigvals, eigvecs = np.linalg.eigh(H)
    gs = eigvecs[:, 0]
    # GS should equal ±1/√8 in every component.
    assert np.allclose(np.abs(gs), 1 / np.sqrt(8), atol=1e-9)


def test_quantum_annealing_preserves_norm():
    """Adiabatic evolution under any Hamiltonian preserves norm."""
    N = 3
    H_init = transverse_field_initial_hamiltonian(N)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    H_prob = _pauli_kron(N, 0, Z) @ _pauli_kron(N, 1, Z)
    psi_0 = np.ones(2 ** N, dtype=complex) / np.sqrt(2 ** N)
    r = quantum_annealing_evolution(H_init, H_prob, psi_0,
                                      n_steps=20, total_time=2.0)
    assert abs(np.linalg.norm(r["final_state"]) - 1.0) < 1e-10


def test_quantum_annealing_finds_ising_ground_state():
    """Slow enough adiabatic sweep on a small Ising problem should
    reach the GS to good accuracy."""
    N = 3
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    H_prob = -(_pauli_kron(N, 0, Z) @ _pauli_kron(N, 1, Z)
                + _pauli_kron(N, 1, Z) @ _pauli_kron(N, 2, Z))
    r = quantum_annealing_ground_state(H_prob, n_steps=400, total_time=30.0)
    assert r["gap_to_true"] < 5e-3


def test_quantum_annealing_diagnostic_output():
    N = 2
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    H_prob = -_pauli_kron(N, 0, Z) @ _pauli_kron(N, 1, Z)
    r = quantum_annealing_ground_state(H_prob, n_steps=50, total_time=5.0)
    assert "energy" in r
    assert "true_gs" in r
    assert "gap_to_true" in r
    assert len(r["energies"]) == 50


def test_quantum_annealing_invalid_dimension():
    H = np.eye(3, dtype=complex)   # not a power of 2
    with pytest.raises(ValueError):
        quantum_annealing_ground_state(H)
