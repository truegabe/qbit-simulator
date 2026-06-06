"""Quantum Boltzmann Machine tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.quantum_boltzmann import (
    qbm_hamiltonian, gibbs_state, computational_basis_distribution,
    empirical_distribution, kl_divergence,
    train_qbm_visible, sample_qbm,
)


# ---- Hamiltonian construction ----

def test_qbm_hamiltonian_hermitian():
    rng = np.random.default_rng(0)
    N = 3
    h = rng.normal(size=N)
    J = np.triu(rng.normal(size=(N, N)), k=1)
    Gamma = rng.normal(size=N)
    H = qbm_hamiltonian(h, J, Gamma)
    assert np.allclose(H, H.conj().T, atol=1e-12)


def test_classical_qbm_is_diagonal():
    """With Γ = 0, the Hamiltonian is diagonal in the computational basis."""
    N = 3
    h = np.ones(N)
    J = np.triu(np.ones((N, N)), k=1)
    Gamma = np.zeros(N)
    H = qbm_hamiltonian(h, J, Gamma)
    off_diag = H - np.diag(np.diag(H))
    assert np.max(np.abs(off_diag)) < 1e-12


def test_qbm_rejects_wrong_shapes():
    h = np.zeros(3)
    J_bad = np.zeros((2, 2))   # wrong shape
    Gamma = np.zeros(3)
    with pytest.raises(ValueError):
        qbm_hamiltonian(h, J_bad, Gamma)


# ---- Gibbs state ----

def test_gibbs_state_is_psd():
    """Gibbs state must be positive semi-definite."""
    H = qbm_hamiltonian(np.array([1, -1, 0.5]),
                          np.array([[0, 0.5, 0.2], [0, 0, -0.3], [0, 0, 0]]),
                          np.array([0.5, 0.3, 0.1]))
    rho = gibbs_state(H, beta=1.0)
    eigs = np.linalg.eigvalsh(rho)
    assert eigs.min() >= -1e-10


def test_gibbs_state_trace_one():
    H = qbm_hamiltonian(np.array([1, -1]),
                         np.array([[0, 0.5], [0, 0]]),
                         np.array([0.2, 0.1]))
    rho = gibbs_state(H, beta=1.5)
    assert abs(np.trace(rho).real - 1.0) < 1e-12


def test_gibbs_state_classical_limit():
    """At β → ∞ on a classical H, the Gibbs state is the GS basis state."""
    H = qbm_hamiltonian(np.array([1.0, 1.0]),
                         np.array([[0, 2.0], [0, 0]]),
                         np.array([0.0, 0.0]))   # classical
    rho = gibbs_state(H, beta=100.0)
    p = computational_basis_distribution(rho)
    # H = -Z_0 - Z_1 - 2 Z_0 Z_1; GS = |00⟩ (Z_0 = +1, Z_1 = +1).
    assert p[0] > 0.99


def test_computational_distribution_sums_to_one():
    H = qbm_hamiltonian(np.array([0.5, -0.3]),
                         np.array([[0, 0.2], [0, 0]]),
                         np.array([0.4, 0.6]))
    rho = gibbs_state(H, beta=2.0)
    p = computational_basis_distribution(rho)
    assert abs(p.sum() - 1.0) < 1e-12
    assert (p >= -1e-12).all()


# ---- KL divergence ----

def test_kl_zero_for_identical():
    p = np.array([0.1, 0.2, 0.3, 0.4])
    assert kl_divergence(p, p) < 1e-9


def test_kl_positive_for_different():
    p = np.array([0.5, 0.5, 0.0, 0.0])
    q = np.array([0.25, 0.25, 0.25, 0.25])
    assert kl_divergence(p, q) > 0


def test_empirical_distribution_normalizes():
    p = empirical_distribution([0, 1, 1, 2, 2, 2], n_bits=2)
    assert abs(p.sum() - 1.0) < 1e-12
    assert abs(p[2] - 3/6) < 1e-12


# ---- Training ----

def test_qbm_trains_on_bimodal():
    """Bimodal target should be learnable (small N, simple problem)."""
    N = 2
    # Target: |00⟩ with prob 0.5, |11⟩ with prob 0.5.
    target = np.array([0.5, 0.0, 0.0, 0.5])
    rng = np.random.default_rng(0)
    result = train_qbm_visible(target, n_visible=N, beta=2.0, lr=0.5,
                                 n_iter=30, rng=rng)
    assert result["kl_final"] < 0.1


def test_qbm_training_reduces_kl():
    target = np.array([0.7, 0.1, 0.1, 0.1])
    rng = np.random.default_rng(0)
    result = train_qbm_visible(target, n_visible=2, beta=1.5, lr=0.3,
                                 n_iter=20, rng=rng)
    assert result["kl_history"][-1] < result["kl_history"][0]


def test_qbm_sampling():
    """Generated samples should follow the model distribution."""
    H = qbm_hamiltonian(np.array([2.0, 2.0]),
                         np.array([[0, 0], [0, 0]]),
                         np.array([0, 0]))
    rho = gibbs_state(H, beta=2.0)
    rng = np.random.default_rng(0)
    samples = sample_qbm(rho, n_samples=2000, rng=rng)
    p_emp = empirical_distribution(samples, n_bits=2)
    p_true = computational_basis_distribution(rho)
    # Empirical should match true distribution.
    for k in range(4):
        assert abs(p_emp[k] - p_true[k]) < 0.05
