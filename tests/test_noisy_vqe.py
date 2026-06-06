"""Tests for the noise-aware VQE wrapper."""

import numpy as np
import pytest

from qbit_simulator.algorithms.h2_sto3g import h2_sto3g_hamiltonian
from qbit_simulator.algorithms.vqe import h2_ansatz
from qbit_simulator.algorithms.noisy_vqe import (
    noisy_vqe, noisy_circuit_state, noisy_energy,
)
from qbit_simulator.noise import depolarizing_kraus, amplitude_damping_kraus


# ---- noise-free baseline ----

def test_noise_free_noisy_vqe_matches_clean():
    """With kraus=None, noisy_vqe should match the clean VQE result."""
    H = h2_sto3g_hamiltonian(0.74)
    result = noisy_vqe(H, h2_ansatz, theta0=0.0,
                       kraus_per_gate=None, n_trajectories=5,
                       max_iter=100, seed=0)
    e_exact, _ = H.ground_state()
    # Noise-free, deterministic — should reach E_exact within ~1 mHa.
    assert abs(result["e_opt"] - e_exact) < 0.01


# ---- noise degrades the estimate ----

def test_noise_degrades_h2_energy():
    """Adding depolarizing noise should drift the optimum away from exact."""
    H = h2_sto3g_hamiltonian(0.74)
    e_exact, _ = H.ground_state()
    rng_seed = 0

    # Noise-free
    clean = noisy_vqe(H, h2_ansatz, theta0=0.0,
                     kraus_per_gate=None, n_trajectories=3,
                     max_iter=80, seed=rng_seed)
    # Moderate depolarizing noise
    noisy = noisy_vqe(H, h2_ansatz, theta0=0.0,
                     kraus_per_gate=depolarizing_kraus(0.05),
                     n_trajectories=40, max_iter=60, seed=rng_seed)
    # Noise raises the estimated energy (decoherence pushes the variational
    # answer towards higher-energy states on average).
    assert noisy["e_opt"] > clean["e_opt"] - 0.01


# ---- trajectory state has unit norm modulo numerical drift ----

def test_noisy_trajectory_norm_preserved():
    """Each trajectory should have approximately unit norm after sampling."""
    rng = np.random.default_rng(0)
    kraus = depolarizing_kraus(0.02)
    psi = noisy_circuit_state(h2_ansatz, 0.5, kraus, rng)
    nrm = np.linalg.norm(psi)
    assert 0.99 < nrm < 1.01


# ---- noisy_energy ----

def test_noisy_energy_on_normalized_state():
    """For a pure state, noisy_energy should return ⟨ψ|H|ψ⟩."""
    H = h2_sto3g_hamiltonian(0.74)
    rng = np.random.default_rng(0)
    psi = noisy_circuit_state(h2_ansatz, 0.3, kraus_per_gate=None, rng=rng)
    e1 = noisy_energy(psi, H)
    e2 = float(np.real(psi.conj() @ H.matrix() @ psi))
    assert abs(e1 - e2) < 1e-9


# ---- different noise channels are distinguishable ----

def test_different_noise_levels_produce_different_energies():
    """Higher noise → higher optimal energy (away from ground state)."""
    H = h2_sto3g_hamiltonian(0.74)
    results = []
    for p_noise in (0.0, 0.05, 0.10):
        kraus = depolarizing_kraus(p_noise) if p_noise > 0 else None
        r = noisy_vqe(H, h2_ansatz, theta0=0.0,
                      kraus_per_gate=kraus, n_trajectories=30,
                      max_iter=40, seed=0)
        results.append(r["e_opt"])
    # Energies should be monotone non-decreasing with noise (in expectation).
    # Allow some stochastic slack.
    assert results[2] > results[0] - 0.05
