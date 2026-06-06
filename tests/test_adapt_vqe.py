"""AdaPT-VQE tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.adapt_vqe import (
    operator_pool_singles_doubles, operator_gradient,
    apply_ansatz, ansatz_energy, adapt_vqe,
)
from qbit_simulator.algorithms.ucc import (
    hartree_fock_state, _pauli_op_to_matrix,
)
from qbit_simulator.fermion import FermionOp, hubbard_hamiltonian


# ---- Operator pool ----

def test_pool_size_h2():
    """4 spin-orbitals, 2 occupied → 4 singles + 1 double = 5."""
    pool = operator_pool_singles_doubles(4, [0, 1])
    assert len(pool) == 5


def test_pool_size_h4():
    """8 spin-orbitals, 4 occupied → 4*4 singles + C(4,2)*C(4,2) doubles."""
    pool = operator_pool_singles_doubles(8, [0, 1, 2, 3])
    n_s = 4 * 4
    n_d = (4 * 3 // 2) * (4 * 3 // 2)
    assert len(pool) == n_s + n_d


# ---- Gradient ----

def test_gradient_at_hf_is_correct_sign():
    """For Hubbard at U=2, the gradient of T_(0,2) at HF should be nonzero."""
    H_pauli = hubbard_hamiltonian(L=2, t=1.0, U=2.0).to_pauli_op(4)
    H_mat = _pauli_op_to_matrix(H_pauli, 4)
    H_mat = 0.5 * (H_mat + H_mat.conj().T)
    psi_hf = hartree_fock_state(4, [0, 1])
    pool = operator_pool_singles_doubles(4, [0, 1])
    grads = [abs(operator_gradient(psi_hf, H_mat, A, 4)) for _, A in pool]
    # At least one gradient should be substantial.
    assert max(grads) > 0.1


def test_gradient_at_ground_state_is_zero():
    """At the exact ground state, all gradients should vanish."""
    H_pauli = hubbard_hamiltonian(L=2, t=1.0, U=2.0).to_pauli_op(4)
    H_mat = _pauli_op_to_matrix(H_pauli, 4)
    H_mat = 0.5 * (H_mat + H_mat.conj().T)
    eigvals, eigvecs = np.linalg.eigh(H_mat)
    psi_gs = eigvecs[:, 0]
    pool = operator_pool_singles_doubles(4, [0, 1])
    grads = [abs(operator_gradient(psi_gs, H_mat, A, 4)) for _, A in pool]
    assert max(grads) < 1e-8


# ---- Apply ansatz ----

def test_empty_ansatz_returns_reference():
    ref = hartree_fock_state(4, [0, 1])
    psi = apply_ansatz([], [], ref, 4)
    assert np.allclose(psi, ref)


def test_ansatz_energy_at_hf_matches_hf_energy():
    H_pauli = hubbard_hamiltonian(L=2, t=1.0, U=2.0).to_pauli_op(4)
    H_mat = _pauli_op_to_matrix(H_pauli, 4)
    ref = hartree_fock_state(4, [0, 1])
    E = ansatz_energy([], H_mat, [], ref, 4)
    # HF for Hubbard 2-site U=2 starting from |↑↓, 0⟩: kinetic=0, U=2.
    assert abs(E - 2.0) < 1e-9


# ---- AdaPT-VQE on a problem where it converges to the exact GS ----

def test_adapt_vqe_solves_hopping_exactly():
    """Single-particle hopping on 2 modes: AdaPT-VQE should reach the
    exact GS (-1.0) in one step."""
    H_ferm = (-FermionOp.cdag(0) * FermionOp.c(1)
              - FermionOp.cdag(1) * FermionOp.c(0))
    H_pauli = H_ferm.to_pauli_op(2)
    result = adapt_vqe(H_pauli, n_qubits=2, occupied=[0],
                       gradient_tol=1e-6, max_iter=5)
    assert abs(result["energy"] - (-1.0)) < 1e-6
    assert result["converged"]
    assert len(result["operators"]) <= 2


# ---- AdaPT-VQE infrastructure properties ----

def test_adapt_vqe_returns_valid_structure():
    H_pauli = hubbard_hamiltonian(L=2, t=1.0, U=2.0).to_pauli_op(4)
    result = adapt_vqe(H_pauli, n_qubits=4, occupied=[0, 1],
                       max_iter=3)
    assert "energy" in result
    assert "thetas" in result
    assert "operators" in result
    assert "gradients" in result
    assert "energies" in result
    assert "converged" in result
    assert len(result["thetas"]) == len(result["operators"])


def test_adapt_vqe_energy_decreases_monotonically():
    """Each iteration of AdaPT-VQE must decrease (or hold) the energy."""
    H_pauli = hubbard_hamiltonian(L=2, t=1.0, U=2.0).to_pauli_op(4)
    result = adapt_vqe(H_pauli, n_qubits=4, occupied=[0, 1],
                       max_iter=5, gradient_tol=1e-12)
    energies = result["energies"]
    for i in range(1, len(energies)):
        assert energies[i] <= energies[i - 1] + 1e-9


def test_adapt_vqe_is_variational():
    """AdaPT-VQE energy ≥ true ground-state energy."""
    H_pauli = hubbard_hamiltonian(L=2, t=1.0, U=2.0).to_pauli_op(4)
    H_mat = _pauli_op_to_matrix(H_pauli, 4)
    true_gs = float(np.linalg.eigvalsh(0.5 * (H_mat + H_mat.conj().T))[0])
    result = adapt_vqe(H_pauli, n_qubits=4, occupied=[0, 1],
                       max_iter=8)
    assert result["energy"] >= true_gs - 1e-9


def test_adapt_vqe_beats_hf():
    """For Hubbard at U=2, AdaPT-VQE should improve over HF energy = 2.0."""
    H_pauli = hubbard_hamiltonian(L=2, t=1.0, U=2.0).to_pauli_op(4)
    result = adapt_vqe(H_pauli, n_qubits=4, occupied=[0, 1],
                       max_iter=5)
    assert result["energy"] < 1.0


def test_adapt_vqe_converges_eventually():
    """The gradient should drop below threshold for an easy problem."""
    H_ferm = -FermionOp.cdag(0) * FermionOp.c(1) - FermionOp.cdag(1) * FermionOp.c(0)
    H_pauli = H_ferm.to_pauli_op(2)
    result = adapt_vqe(H_pauli, n_qubits=2, occupied=[0],
                       gradient_tol=1e-4, max_iter=10)
    assert result["converged"]
