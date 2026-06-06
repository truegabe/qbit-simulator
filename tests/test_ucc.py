"""UCCSD ansatz tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.ucc import (
    single_excitation, double_excitation,
    singles_generators, doubles_generators, uccsd_generators,
    n_parameters, hartree_fock_state, uccsd_ansatz, uccsd_energy,
    apply_excitation, _generator_as_hermitian_matrix,
)


# ---- Excitation generators ----

def test_single_excitation_antihermitian():
    """T_ia = c†_a c_i − c†_i c_a should give a Hermitian H = i · JW(T_ia)."""
    G = single_excitation(0, 2)
    H = _generator_as_hermitian_matrix(G, n_modes=4)
    assert np.allclose(H, H.conj().T, atol=1e-12)


def test_double_excitation_antihermitian():
    G = double_excitation(0, 1, 2, 3)
    H = _generator_as_hermitian_matrix(G, n_modes=4)
    assert np.allclose(H, H.conj().T, atol=1e-12)


def test_singles_count_h2():
    """H2 (4 spin-orbitals, 2 occupied): C(2,1)·C(2,1) = 4 singles."""
    assert len(singles_generators(4, [0, 1])) == 4


def test_doubles_count_h2():
    """H2: C(2,2)·C(2,2) = 1 double."""
    assert len(doubles_generators(4, [0, 1])) == 1


def test_n_parameters_h2():
    p = n_parameters(4, [0, 1])
    assert p["singles"] == 4
    assert p["doubles"] == 1
    assert p["total"] == 5


# ---- Hartree-Fock state ----

def test_hartree_fock_normalized():
    psi = hartree_fock_state(4, [0, 1])
    assert abs(np.linalg.norm(psi) - 1.0) < 1e-12


def test_hartree_fock_is_basis_state():
    """|HF⟩ should be a single computational basis state."""
    psi = hartree_fock_state(4, [0, 1])
    # Exactly one entry is 1, rest are 0.
    abs_psi = np.abs(psi)
    assert np.sum(abs_psi > 1e-10) == 1


def test_hartree_fock_correct_occupation():
    """Modes 0 and 1 occupied → MSB-first |1100⟩ → index 12."""
    psi = hartree_fock_state(4, [0, 1])
    assert psi[12] == 1.0
    assert abs(psi.sum() - 1.0) < 1e-12   # only one entry nonzero


# ---- UCCSD ansatz properties ----

def test_uccsd_zero_thetas_equals_hf():
    psi_hf = hartree_fock_state(4, [0, 1])
    psi_zero = uccsd_ansatz([0.0]*5, 4, [0, 1])
    assert np.allclose(psi_hf, psi_zero, atol=1e-12)


def test_uccsd_state_is_normalized():
    rng = np.random.default_rng(0)
    for _ in range(5):
        thetas = rng.uniform(-1, 1, size=5).tolist()
        psi = uccsd_ansatz(thetas, 4, [0, 1])
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-9


def test_uccsd_preserves_particle_number():
    """All excitation operators conserve fermion number."""
    rng = np.random.default_rng(0)
    thetas = rng.uniform(-0.5, 0.5, size=5).tolist()
    psi = uccsd_ansatz(thetas, 4, [0, 1])
    # Particle number operator N = sum_k (I - Z_k) / 2.
    # We just check that the state lies in the 2-particle subspace
    # (basis states with exactly 2 bits set).
    for k in range(16):
        if abs(psi[k]) > 1e-9:
            n_bits = bin(k).count("1")
            assert n_bits == 2


def test_apply_excitation_unitary():
    G = single_excitation(0, 2)
    psi = hartree_fock_state(4, [0, 1])
    psi_new = apply_excitation(psi, G, theta=0.5, n_qubits=4)
    assert abs(np.linalg.norm(psi_new) - 1.0) < 1e-12


# ---- UCCSD energy ----

def test_uccsd_energy_at_zero_is_hf_energy():
    from qbit_simulator.fermion import hubbard_hamiltonian
    H = hubbard_hamiltonian(L=2, t=1.0, U=2.0).to_pauli_op(4)
    E_zero = uccsd_energy([0.0]*5, H, 4, [0, 1])
    # For HF |↑↓, 0⟩ in 2-site Hubbard with U=2: kinetic=0, interaction=U=2.
    assert abs(E_zero - 2.0) < 1e-9


def test_uccsd_energy_variational():
    """UCCSD energy ≥ true ground-state energy for any theta."""
    from qbit_simulator.fermion import hubbard_hamiltonian
    from qbit_simulator.algorithms.ucc import _pauli_op_to_matrix
    H = hubbard_hamiltonian(L=2, t=1.0, U=2.0).to_pauli_op(4)
    H_mat = _pauli_op_to_matrix(H, 4)
    E_gs = float(np.linalg.eigvalsh(H_mat)[0])
    rng = np.random.default_rng(0)
    for _ in range(10):
        thetas = rng.uniform(-1, 1, size=5).tolist()
        E = uccsd_energy(thetas, H, 4, [0, 1])
        assert E >= E_gs - 1e-9


def test_uccsd_lowers_energy_below_hf():
    """UCCSD with optimized parameters should beat HF for Hubbard."""
    from qbit_simulator.fermion import hubbard_hamiltonian
    from scipy.optimize import minimize
    H = hubbard_hamiltonian(L=2, t=1.0, U=2.0).to_pauli_op(4)

    def loss(thetas):
        return uccsd_energy(list(thetas), H, 4, [0, 1])

    rng = np.random.default_rng(0)
    best = float("inf")
    for _ in range(5):
        x0 = rng.uniform(-0.5, 0.5, size=5)
        res = minimize(loss, x0, method="BFGS", options={"maxiter": 200})
        if res.fun < best:
            best = res.fun
    E_hf = 2.0   # HF energy from above
    assert best < E_hf - 0.5


def test_apply_excitation_rotates_in_hf_orthogonal_subspace():
    """exp(θ · T_ia) creates a state with cos(θ)|HF⟩ + sin(θ)·(excited state)."""
    G = double_excitation(0, 1, 2, 3)
    psi_hf = hartree_fock_state(4, [0, 1])
    psi = apply_excitation(psi_hf, G, theta=np.pi/4, n_qubits=4)
    # Overlap with |HF⟩ should be cos(theta·something) for some "something".
    overlap = abs(np.vdot(psi, psi_hf))
    # Not 1 (we rotated away), not 0 (didn't fully flip).
    assert 0.1 < overlap < 0.99
