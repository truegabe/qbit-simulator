"""Lanczos + Davidson eigensolver tests."""

import numpy as np
import pytest

from qbit_simulator.lanczos import (
    lanczos_iterate, lanczos_ground_state, lanczos_lowest_k,
    davidson_ground_state,
    matvec_from_pauli_op, _apply_pauli_string_to_vec,
)


# ---- Pauli matvec helper ----

def test_apply_pauli_z_on_basis_state():
    """Z on |0⟩ gives +|0⟩, on |1⟩ gives -|1⟩."""
    v = np.array([1, 0], dtype=complex)
    out = _apply_pauli_string_to_vec("Z", v, n=1)
    assert np.allclose(out, [1, 0])
    v = np.array([0, 1], dtype=complex)
    out = _apply_pauli_string_to_vec("Z", v, n=1)
    assert np.allclose(out, [0, -1])


def test_apply_pauli_x_flips():
    v = np.array([1, 0], dtype=complex)
    out = _apply_pauli_string_to_vec("X", v, n=1)
    assert np.allclose(out, [0, 1])


def test_apply_pauli_y_on_zero():
    """Y |0⟩ = i|1⟩."""
    v = np.array([1, 0], dtype=complex)
    out = _apply_pauli_string_to_vec("Y", v, n=1)
    assert np.allclose(out, [0, 1j])


def test_apply_pauli_string_xz():
    """X⊗Z on |11⟩ = |01⟩ · (-1)."""
    v = np.array([0, 0, 0, 1], dtype=complex)
    out = _apply_pauli_string_to_vec("XZ", v, n=2)
    # |11⟩ → X on q0: |01⟩ (idx 1). Then Z on q1: bit 1 is 1, so -|01⟩.
    assert np.allclose(out, [0, -1, 0, 0])


# ---- Lanczos iteration ----

def test_lanczos_iterate_basis_orthonormal():
    n = 16
    rng = np.random.default_rng(0)
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    H = (A + A.conj().T) / 2
    alphas, betas, basis = lanczos_iterate(lambda v: H @ v, n, k_dim=8, rng=rng)
    # Basis vectors should be orthonormal.
    for i, u in enumerate(basis):
        assert abs(np.linalg.norm(u) - 1.0) < 1e-8
        for j, v in enumerate(basis[:i]):
            assert abs(np.vdot(u, v)) < 1e-8


def test_lanczos_alphas_betas_lengths():
    n = 8
    rng = np.random.default_rng(0)
    H = np.eye(n, dtype=complex)
    alphas, betas, basis = lanczos_iterate(
        lambda v: H @ v, n, k_dim=5, rng=rng,
    )
    # Identity: span(v_0) is invariant, so betas should be 0 immediately.
    # We expect k=1 (one alpha, no betas, one basis vector).
    assert len(alphas) == 1
    assert len(basis) == 1


# ---- Lanczos ground state ----

def test_lanczos_ground_state_random_hermitian():
    """Lanczos should find the lowest eigenvalue of a random Hermitian."""
    rng = np.random.default_rng(0)
    n = 32
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    H = (A + A.conj().T) / 2
    true_min = float(np.linalg.eigvalsh(H)[0])
    r = lanczos_ground_state(lambda v: H @ v, n, k_dim=20, rng=rng)
    assert abs(r["energy"] - true_min) < 1e-3


def test_lanczos_with_pauli_op():
    """Lanczos on a Pauli-Op Hamiltonian should match exact diag."""
    from qbit_simulator.algorithms.h2_sto3g import (
        h2_sto3g_hamiltonian, h2_sto3g_energy,
    )
    H_p = h2_sto3g_hamiltonian(0.74)
    mv = matvec_from_pauli_op(H_p, n_qubits=2)
    rng = np.random.default_rng(0)
    r = lanczos_ground_state(mv, n=4, k_dim=10, rng=rng)
    true_gs = h2_sto3g_energy(0.74)
    assert abs(r["energy"] - true_gs) < 1e-6


def test_lanczos_returns_eigenvector():
    rng = np.random.default_rng(0)
    n = 8
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    H = (A + A.conj().T) / 2
    r = lanczos_ground_state(lambda v: H @ v, n, k_dim=8, rng=rng)
    v = r["eigenvector"]
    assert v.shape == (n,)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-8


def test_lanczos_eigenvector_satisfies_eigenequation():
    """H · v ≈ E · v for the recovered eigenvector."""
    rng = np.random.default_rng(0)
    n = 16
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    H = (A + A.conj().T) / 2
    r = lanczos_ground_state(lambda v: H @ v, n, k_dim=16, rng=rng)
    Hv = H @ r["eigenvector"]
    residual = np.linalg.norm(Hv - r["energy"] * r["eigenvector"])
    assert residual < 1e-3


# ---- Lowest-k ----

def test_lanczos_lowest_k_returns_k_values():
    rng = np.random.default_rng(0)
    n = 16
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    H = (A + A.conj().T) / 2
    r = lanczos_lowest_k(lambda v: H @ v, n, n_eigs=3, k_dim=10, rng=rng)
    assert len(r["eigenvalues"]) == 3
    assert len(r["eigenvectors"]) == 3


def test_lanczos_lowest_k_sorted_ascending():
    rng = np.random.default_rng(0)
    n = 16
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    H = (A + A.conj().T) / 2
    r = lanczos_lowest_k(lambda v: H @ v, n, n_eigs=3, k_dim=12, rng=rng)
    eigs = r["eigenvalues"]
    for i in range(len(eigs) - 1):
        assert eigs[i] <= eigs[i + 1]


# ---- Davidson ----

def test_davidson_ground_state_runs():
    """Davidson on a random Hermitian; check basic structure."""
    rng = np.random.default_rng(0)
    n = 16
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    H = (A + A.conj().T) / 2
    diag = np.diag(H).real
    r = davidson_ground_state(lambda v: H @ v, diag, n, max_subspace=15, rng=rng)
    assert "energy" in r
    assert "eigenvector" in r
    assert r["eigenvector"].shape == (n,)


def test_davidson_finds_minimum_eigenvalue():
    """Davidson should produce an eigenvalue near the minimum."""
    rng = np.random.default_rng(0)
    n = 16
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    H = (A + A.conj().T) / 2
    true_min = float(np.linalg.eigvalsh(H)[0])
    diag = np.diag(H).real
    r = davidson_ground_state(lambda v: H @ v, diag, n,
                                max_subspace=15, rng=rng)
    assert r["energy"] - true_min < 0.1   # within 10%


def test_davidson_returns_normalized_eigenvector():
    rng = np.random.default_rng(0)
    n = 8
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    H = (A + A.conj().T) / 2
    diag = np.diag(H).real
    r = davidson_ground_state(lambda v: H @ v, diag, n, max_subspace=8, rng=rng)
    assert abs(np.linalg.norm(r["eigenvector"]) - 1.0) < 1e-8


# ---- Matvec adapter ----

def test_matvec_from_pauli_op_matches_dense():
    """The Pauli-Op-based matvec should equal the dense matrix times v."""
    from qbit_simulator.algorithms.h2_sto3g import h2_sto3g_hamiltonian
    from qbit_simulator.algorithms.ucc import _pauli_op_to_matrix
    H_p = h2_sto3g_hamiltonian(0.74)
    H_mat = _pauli_op_to_matrix(H_p, 2)
    mv = matvec_from_pauli_op(H_p, n_qubits=2)
    rng = np.random.default_rng(0)
    v = rng.normal(size=4) + 1j * rng.normal(size=4)
    out_dense = H_mat @ v
    out_mv = mv(v)
    assert np.allclose(out_dense, out_mv, atol=1e-9)
