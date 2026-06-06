"""Jordan-Wigner and fermion-operator tests."""

import numpy as np
import pytest

from qbit_simulator.fermion import (
    FermionOp, hubbard_hamiltonian, _pauli_string_mul,
)


# ---- Pauli multiplication primitive ----

def test_pauli_mul_table():
    assert _pauli_string_mul("X", "X") == (1+0j, "I")
    assert _pauli_string_mul("X", "Y") == (1j, "Z")
    assert _pauli_string_mul("Y", "X") == (-1j, "Z")
    assert _pauli_string_mul("Z", "Z") == (1+0j, "I")
    assert _pauli_string_mul("X", "Z") == (-1j, "Y")
    assert _pauli_string_mul("Z", "X") == (1j, "Y")


def test_pauli_mul_strings():
    c, s = _pauli_string_mul("XY", "YX")
    # (X*Y) on q0 = iZ ; (Y*X) on q1 = -iZ.  Product: -i * i = -i^2 = 1.
    assert s == "ZZ"
    assert abs(c - 1) < 1e-12


# ---- Fermion operator basics ----

def test_c_and_cdag_are_distinct():
    c0 = FermionOp.c(0)
    cd0 = FermionOp.cdag(0)
    assert c0.terms != cd0.terms


def test_number_op_squared():
    """n_0² = n_0 (for fermions: n is idempotent)."""
    n0 = FermionOp.number(0)
    n_sq = n0 * n0
    # After JW, both should produce the same PauliOp.
    # n_0 = (I - Z_0)/2; n_0² = (I - 2Z_0 + Z_0²)/4 = (I - 2Z_0 + I)/4 = (I - Z_0)/2 = n_0.
    p1 = n0.to_pauli_op(2)
    p2 = n_sq.to_pauli_op(2)
    # Compare matrices.
    assert np.allclose(p1.matrix(), p2.matrix(), atol=1e-10)


def test_canonical_anticommutation():
    """{c_i, c†_i} = c_i c†_i + c†_i c_i = I."""
    c0 = FermionOp.c(0)
    cd0 = FermionOp.cdag(0)
    anticomm = c0 * cd0 + cd0 * c0
    M = anticomm.to_pauli_op(2).matrix()
    assert np.allclose(M, np.eye(4), atol=1e-10)


def test_anticommutation_distinct_modes():
    """{c_i, c_j} = 0 for i ≠ j."""
    c0 = FermionOp.c(0)
    c1 = FermionOp.c(1)
    anticomm = c0 * c1 + c1 * c0
    M = anticomm.to_pauli_op(2).matrix()
    assert np.allclose(M, np.zeros((4, 4)), atol=1e-10)


def test_creation_squared_is_zero():
    """c†_i² = 0 (Pauli exclusion)."""
    cd0 = FermionOp.cdag(0)
    M = (cd0 * cd0).to_pauli_op(2).matrix()
    assert np.allclose(M, np.zeros((4, 4)), atol=1e-10)


# ---- Number operator action ----

def test_number_op_on_basis_states():
    """n_0 acting on |0⟩, |1⟩ gives 0, 1 respectively."""
    n0 = FermionOp.number(0)
    M = n0.to_pauli_op(2).matrix()
    # State |00⟩: indices 0..3 with binary 00 = 0 -> qubit 0 is 0 (MSB convention)
    # Actually our convention: qubit 0 is MSB. n_0 = (I - Z_0)/2 in JW.
    # |00⟩ in qubit-0-MSB convention is index 0. Bit at qubit 0 = 0.
    # We want n_0|x⟩ = x_0 |x⟩ where x_0 is the bit at qubit 0.
    # So M|00⟩ = 0, M|10⟩ = 1·|10⟩, etc.
    for i in range(4):
        bit0 = (i >> 1) & 1   # qubit 0 is bit 1 of the index (MSB of 2-bit)
        e_i = np.zeros(4); e_i[i] = 1
        result = M @ e_i
        # n_0 |i⟩ = bit0 · |i⟩
        assert np.allclose(result, bit0 * e_i), \
            f"n_0|{i:02b}⟩ = {result}, expected {bit0 * e_i}"


# ---- Hubbard model ----

def test_hubbard_dimer_construction():
    """Hubbard dimer (L=2) has 4 spin-orbitals."""
    H = hubbard_hamiltonian(L=2, t=1.0, U=4.0)
    assert H.n_modes() == 4
    pH = H.to_pauli_op(4)
    M = pH.matrix()
    # 16x16 Hermitian matrix.
    assert M.shape == (16, 16)
    assert np.allclose(M, M.conj().T, atol=1e-10)


def test_hubbard_dimer_half_filling_ground_in_correct_sector():
    """At half filling (N_e = 2 on L=2 sites) and t=1, U=4, the Hubbard
    dimer's ground energy is

        E_half = (U - √(U² + 16t²))/2

    For U=4, t=1: E_half = 2 - 2√2 ≈ -0.8284

    We project the full Hamiltonian onto the N_e = 2 particle-number
    sector and verify the ground energy there matches the analytic
    Bethe-ansatz value. (Without projection, the unconstrained ground
    is the empty-orbital vacuum at E=0.)
    """
    from qbit_simulator.fermion import project_to_particle_number_sector
    H = hubbard_hamiltonian(L=2, t=1.0, U=4.0)
    pH = H.to_pauli_op(4)
    # Project onto the 2-electron sector (half-filling of 2 sites = 2 e-).
    H_half = project_to_particle_number_sector(pH.matrix(), n_target=2)
    e_gs = float(np.linalg.eigvalsh(H_half)[0])
    expected = 2.0 - 2 * np.sqrt(2)
    assert abs(e_gs - expected) < 1e-9


def test_particle_number_sector_dimensions():
    """For L=2 (4 spin-orbitals), the N-particle sectors have dim C(4, N)."""
    from math import comb
    from qbit_simulator.fermion import project_to_particle_number_sector
    H = hubbard_hamiltonian(L=2, t=1.0, U=4.0).to_pauli_op(4).matrix()
    for N in range(5):
        H_sector = project_to_particle_number_sector(H, n_target=N)
        assert H_sector.shape == (comb(4, N), comb(4, N))


def test_hubbard_chain_3_sites():
    """A 3-site Hubbard chain has 6 spin-orbitals and a well-defined ground state."""
    H = hubbard_hamiltonian(L=3, t=1.0, U=2.0)
    pH = H.to_pauli_op(6)
    e_gs, gs = pH.ground_state()
    # The ground state is real and negative (kinetic energy dominates over U=2t).
    assert e_gs < 0
    # State is normalized.
    assert abs(np.linalg.norm(gs) - 1.0) < 1e-9
