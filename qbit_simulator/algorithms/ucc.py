"""Unitary Coupled Cluster (UCC) ansatz for quantum chemistry.

The UCC ansatz is a chemistry-tailored alternative to hardware-efficient
ansätze in VQE. It's parameterized by single- and double-excitation
amplitudes from a Hartree-Fock reference:

    |ψ(θ)⟩ = exp(T(θ) − T†(θ)) |HF⟩

with

    T  = T1 + T2 + …
    T1 = sum_{i,a} θ_ia c†_a c_i           (single excitations)
    T2 = sum_{ijab} θ_ijab c†_a c†_b c_j c_i   (double excitations)

where i, j range over OCCUPIED orbitals (in |HF⟩) and a, b over VIRTUAL
orbitals. The operator (T − T†) is anti-Hermitian, so exp(T − T†) is
unitary.

In practice we use the **Trotterized UCC**: exp(sum_k τ_k) ≈ prod_k
exp(τ_k). Each generator τ_k = (c†_a c_i − c†_i c_a) maps under Jordan-
Wigner to a sum of Pauli strings; we exponentiate each Pauli string
individually.

This module provides:

  - `singles_generators(n_orbitals, occupied)`: list of T_ia generators
  - `doubles_generators(n_orbitals, occupied)`: list of T_ijab generators
  - `uccsd_generators(n_orbitals, occupied)`: union of T1 and T2
  - `apply_excitation(state, generator, theta)`: apply exp(theta · G)
    to a state vector by exponentiating each Pauli term of G.
  - `uccsd_ansatz(thetas, n_qubits, occupied)`: build the full UCCSD
    circuit on a state vector.

For H₂/STO-3G (2 occupied + 2 virtual = 4 spin-orbitals = 4 qubits),
UCCSD has 2 singles + 1 double = 3 parameters and recovers the exact
FCI energy with VQE.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from ..fermion import FermionOp
from ..pauli import PauliOp


# ----------------------------------------------------------------------------
# Excitation generators
# ----------------------------------------------------------------------------

def single_excitation(i: int, a: int) -> FermionOp:
    """T_ia = c†_a c_i − c†_i c_a  (anti-Hermitian single-excitation generator).

    Excites one electron from orbital i to orbital a.
    """
    return FermionOp.cdag(a) * FermionOp.c(i) - FermionOp.cdag(i) * FermionOp.c(a)


def double_excitation(i: int, j: int, a: int, b: int) -> FermionOp:
    """T_ijab = c†_a c†_b c_j c_i − c†_i c†_j c_b c_a  (anti-Hermitian).

    Excites two electrons from orbitals (i, j) to (a, b).
    """
    forward = (FermionOp.cdag(a) * FermionOp.cdag(b)
               * FermionOp.c(j) * FermionOp.c(i))
    backward = (FermionOp.cdag(i) * FermionOp.cdag(j)
                * FermionOp.c(b) * FermionOp.c(a))
    return forward - backward


def singles_generators(n_orbitals: int, occupied: list[int]
                        ) -> list[tuple[tuple[int, int], FermionOp]]:
    """All occupied→virtual single-excitation generators.

    Returns a list of ((i, a), T_ia) where i ∈ occupied, a ∈ virtual.
    """
    occ_set = set(occupied)
    virtual = [k for k in range(n_orbitals) if k not in occ_set]
    out = []
    for i in occupied:
        for a in virtual:
            out.append(((i, a), single_excitation(i, a)))
    return out


def doubles_generators(n_orbitals: int, occupied: list[int]
                        ) -> list[tuple[tuple[int, int, int, int], FermionOp]]:
    """All (i<j) occupied → (a<b) virtual double-excitation generators."""
    occ_set = set(occupied)
    virtual = [k for k in range(n_orbitals) if k not in occ_set]
    out = []
    for (i, j) in combinations(occupied, 2):
        for (a, b) in combinations(virtual, 2):
            out.append(((i, j, a, b), double_excitation(i, j, a, b)))
    return out


def uccsd_generators(n_orbitals: int, occupied: list[int]
                      ) -> list[tuple[tuple, FermionOp]]:
    """All UCCSD excitation generators (singles + doubles)."""
    return (singles_generators(n_orbitals, occupied)
            + doubles_generators(n_orbitals, occupied))


# ----------------------------------------------------------------------------
# Apply exp(theta · G) to a state vector
# ----------------------------------------------------------------------------

def _pauli_string_to_matrix(s: str) -> np.ndarray:
    """Build the dense Hilbert-space matrix for a Pauli string."""
    _I = np.eye(2, dtype=np.complex128)
    _X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    _Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    _Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    table = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}
    M = np.array([[1.0 + 0j]])
    # JW convention: leftmost char = mode 0, but kron uses qubit 0 as MSB.
    # We follow the fermion-module convention (qubit k corresponds to
    # position k in the Pauli string).
    for ch in s:
        M = np.kron(M, table[ch])
    return M


def _generator_as_hermitian_matrix(G: FermionOp, n_modes: int) -> np.ndarray:
    """Build the dense matrix for the JW image of G, multiplied by i so the
    result is Hermitian (since G is anti-Hermitian).

    Returns a 2^n_modes × 2^n_modes complex matrix.
    """
    pauli = G.to_pauli_op(n_modes)
    dim = 2 ** n_modes
    M = np.zeros((dim, dim), dtype=np.complex128)
    for coef, s in pauli.terms:
        # G is anti-Hermitian, so JW image is i * (Hermitian).
        # We absorb the i: H = i · JW(G).
        # Then exp(θ · G) = exp(-i · θ · H).
        M = M + coef * _pauli_string_to_matrix(s)
    # Result should be anti-Hermitian: M = -M†.
    # We return i*M (which is then Hermitian).
    H = 1j * M
    # Numerical hermitization.
    H = 0.5 * (H + H.conj().T)
    return H


def apply_excitation(psi: np.ndarray, generator: FermionOp,
                      theta: float, n_qubits: int) -> np.ndarray:
    """Apply exp(theta · generator) to a state vector via exact matrix exp.

    For UCC, generator is anti-Hermitian so this is unitary.
    """
    H = _generator_as_hermitian_matrix(generator, n_qubits)
    # exp(theta · G) = exp(-i · theta · H) where H = i · JW(G).
    # Use eigendecomposition (H is small for chemistry-size systems).
    eigs, V = np.linalg.eigh(H)
    U = V @ np.diag(np.exp(-1j * theta * eigs)) @ V.conj().T
    return U @ psi


# ----------------------------------------------------------------------------
# Reference state and full ansatz
# ----------------------------------------------------------------------------

def hartree_fock_state(n_qubits: int, occupied: list[int]) -> np.ndarray:
    """Build |HF⟩ = |occupied⟩ as a computational-basis state vector.

    In JW the occupation pattern n_0 n_1 … n_{N-1} maps to the basis
    state with the corresponding bit pattern (with our convention that
    mode k corresponds to the k-th qubit, indexed from 0 = MSB in the
    Pauli string).
    """
    dim = 2 ** n_qubits
    # The basis index whose binary representation has 1s at occupied positions.
    idx = 0
    for k in occupied:
        idx |= (1 << (n_qubits - 1 - k))   # qubit k = position from MSB
    psi = np.zeros(dim, dtype=np.complex128)
    psi[idx] = 1.0
    return psi


def uccsd_ansatz(thetas: list[float],
                  n_qubits: int,
                  occupied: list[int]) -> np.ndarray:
    """Build |ψ(θ)⟩ = (Trotterized UCCSD on |HF⟩).

    Args:
        thetas:    list of variational parameters, one per generator.
        n_qubits:  number of spin-orbitals (qubits).
        occupied:  list of occupied-orbital indices in |HF⟩.

    Returns:
        the 2^n_qubits state vector.
    """
    gens = uccsd_generators(n_qubits, occupied)
    if len(thetas) != len(gens):
        raise ValueError(f"expected {len(gens)} parameters, got {len(thetas)}")
    psi = hartree_fock_state(n_qubits, occupied)
    # Trotterized: apply each exp(theta_k · G_k) in turn.
    for theta, (_, G) in zip(thetas, gens):
        psi = apply_excitation(psi, G, float(theta), n_qubits)
    return psi


# ----------------------------------------------------------------------------
# VQE energy
# ----------------------------------------------------------------------------

def uccsd_energy(thetas: list[float],
                  hamiltonian: PauliOp,
                  n_qubits: int,
                  occupied: list[int]) -> float:
    """Compute ⟨ψ(θ) | H | ψ(θ)⟩ for the UCCSD ansatz."""
    psi = uccsd_ansatz(thetas, n_qubits, occupied)
    H_mat = _pauli_op_to_matrix(hamiltonian, n_qubits)
    return float(np.real(psi.conj() @ H_mat @ psi))


def _pauli_op_to_matrix(op: PauliOp, n: int) -> np.ndarray:
    dim = 2 ** n
    M = np.zeros((dim, dim), dtype=np.complex128)
    for coef, s in op.terms:
        M = M + coef * _pauli_string_to_matrix(s)
    return M


# ----------------------------------------------------------------------------
# Counting
# ----------------------------------------------------------------------------

def n_parameters(n_orbitals: int, occupied: list[int]) -> dict:
    """Count UCCSD parameters: singles, doubles, total."""
    n_s = len(singles_generators(n_orbitals, occupied))
    n_d = len(doubles_generators(n_orbitals, occupied))
    return {"singles": n_s, "doubles": n_d, "total": n_s + n_d}
