"""Schwinger model (1+1 D QED) on a staggered lattice.

The Schwinger model is quantum electrodynamics in (1+1) dimensions. It's
the simplest lattice gauge theory that exhibits genuine non-trivial
physics:

  * Confinement: oppositely-charged fermions are bound by a linear
    "string" potential (analogous to quarks in QCD).
  * A topological θ-term and string breaking when fermion pairs are
    created spontaneously from the vacuum.

After integrating out the gauge field (in 1+1D this is possible because
A_0 is a Lagrange multiplier), the Hamiltonian on a staggered lattice
of N sites reduces to a pure-fermion system:

    H = -i w · sum_n (φ†_n φ_{n+1} − φ†_{n+1} φ_n)        (hopping)
        + m · sum_n (-1)^n φ†_n φ_n                       (staggered mass)
        + (g²/2) · sum_n L_n²                              (electric energy)

The electric field L_n is determined non-locally by Gauss's law:

    L_n = ε_0 + sum_{k ≤ n} (φ†_k φ_k − (1 − (-1)^k)/2)

with ε_0 the background field (allows the θ-term: θ = 2π ε_0).

This module provides:

  - `schwinger_hamiltonian(N, w, m, g, eps0)`: build the spin-1/2 Pauli
    Hamiltonian via Jordan-Wigner from the lattice fermion operators.
  - `schwinger_ground_state(...)`: exact diagonalization for small N.
  - `chiral_condensate(state, N)`: <φ̄ φ> — order parameter for chiral
    symmetry breaking.
  - `string_tension_demo(...)`: measure the string tension between two
    test charges (illustrates confinement).

For N ≤ 8 sites we use exact diagonalization in the full 2^N Hilbert
space. (Going to larger N would require sparse methods or DMRG.)
"""

from __future__ import annotations

import numpy as np

from ..fermion import FermionOp
from ..pauli import PauliOp


# ----------------------------------------------------------------------------
# Build the Hamiltonian
# ----------------------------------------------------------------------------

def schwinger_hamiltonian(
    N: int,
    w: float = 1.0,    # hopping strength
    m: float = 0.5,    # staggered mass
    g: float = 1.0,    # gauge coupling
    eps0: float = 0.0,  # background electric field (theta-angle / 2pi)
) -> PauliOp:
    """Schwinger-model Hamiltonian on a staggered lattice of N sites
    after eliminating the gauge field by Gauss's law.

    Returns a PauliOp on N qubits.
    """
    if N < 2:
        raise ValueError("need N >= 2 sites")

    H = FermionOp.zero()

    # Hopping term: -i w · sum_n (φ†_n φ_{n+1} - h.c.)
    # = -i w · (φ†_n φ_{n+1} - φ†_{n+1} φ_n)
    # Open boundary: site index n = 0, ..., N-2 for hopping.
    for n in range(N - 1):
        term = FermionOp.cdag(n) * FermionOp.c(n + 1)
        H = H + (-1j * w) * term
        H = H + (1j * w) * (FermionOp.cdag(n + 1) * FermionOp.c(n))

    # Staggered mass term: m · sum_n (-1)^n n_n
    for n in range(N):
        H = H + (m * ((-1) ** n)) * FermionOp.number(n)

    # Electric-field energy: (g²/2) · sum_{n=0}^{N-2} L_n²
    # where L_n = ε_0 + sum_{k=0}^{n} (n_k - (1 - (-1)^k)/2)
    # The "static charge density" baseline 1 - (-1)^k = 2 if k odd, 0 if even.
    # So baseline = 1 if k odd, 0 if k even. We subtract this from n_k.
    for n in range(N - 1):
        # Build L_n as a FermionOp.
        L_n = eps0 * FermionOp.identity()
        for k in range(n + 1):
            baseline = 1.0 if k % 2 == 1 else 0.0
            L_n = L_n + FermionOp.number(k)
            L_n = L_n + (-baseline) * FermionOp.identity() if baseline else L_n
        H = H + (0.5 * g ** 2) * (L_n * L_n)

    return H.to_pauli_op(N)


def schwinger_hamiltonian_matrix(N: int, **kwargs) -> np.ndarray:
    """Dense matrix form of the Schwinger Hamiltonian (for small N)."""
    H_pauli = schwinger_hamiltonian(N, **kwargs)
    dim = 2 ** N
    M = np.zeros((dim, dim), dtype=np.complex128)
    for coef, s in H_pauli.terms:
        M = M + coef * _pauli_string_matrix(s)
    return 0.5 * (M + M.conj().T)   # numerical hermitization


def _pauli_string_matrix(s: str) -> np.ndarray:
    _I = np.eye(2, dtype=np.complex128)
    _X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    _Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    _Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    table = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}
    M = np.array([[1.0 + 0j]])
    for ch in s:
        M = np.kron(M, table[ch])
    return M


# ----------------------------------------------------------------------------
# Ground state and observables
# ----------------------------------------------------------------------------

def schwinger_ground_state(N: int, **kwargs) -> tuple[float, np.ndarray]:
    """Lowest eigenvalue and eigenstate by exact diagonalization."""
    H = schwinger_hamiltonian_matrix(N, **kwargs)
    eigvals, eigvecs = np.linalg.eigh(H)
    return float(eigvals[0]), eigvecs[:, 0]


def chiral_condensate(psi: np.ndarray, N: int) -> float:
    """⟨ψ̄ ψ⟩ — the chiral condensate order parameter for chiral symmetry
    breaking.

    On the staggered lattice:
        ⟨ψ̄ ψ⟩ = (1/N) sum_n (-1)^n ⟨n_n⟩  =  (1/N) sum_n (-1)^n ⟨(I - Z_n)/2⟩

    Nonzero condensate = chiral symmetry broken (the model has a
    non-trivial vacuum even at m → 0 due to the chiral anomaly).
    """
    if 2 ** N != len(psi):
        raise ValueError(f"state length {len(psi)} != 2^{N}")
    total = 0.0
    for n in range(N):
        # ⟨n_n⟩ = ⟨(I - Z_n)/2⟩
        s = "I" * n + "Z" + "I" * (N - 1 - n)
        Z_n = _pauli_string_matrix(s)
        exp_Z = float(np.real(psi.conj() @ Z_n @ psi))
        n_avg = 0.5 * (1 - exp_Z)
        total += ((-1) ** n) * n_avg
    return total / N


def electric_field_per_link(psi: np.ndarray, N: int, eps0: float = 0.0
                              ) -> np.ndarray:
    """⟨L_n⟩ on each link n = 0, ..., N-2.

    L_n = ε_0 + sum_{k≤n} (n_k - baseline_k) with baseline_k = 1 if k odd.
    """
    L = np.zeros(N - 1)
    for n in range(N - 1):
        total = eps0
        for k in range(n + 1):
            baseline = 1.0 if k % 2 == 1 else 0.0
            s = "I" * k + "Z" + "I" * (N - 1 - k)
            Z_k = _pauli_string_matrix(s)
            exp_Z = float(np.real(psi.conj() @ Z_k @ psi))
            n_avg = 0.5 * (1 - exp_Z)
            total += n_avg - baseline
        L[n] = total
    return L


def total_charge(psi: np.ndarray, N: int) -> float:
    """⟨Q⟩ = sum_n ⟨n_n - baseline⟩ — total fermion charge above
    background.

    For the Schwinger vacuum (the lowest-energy state with ε_0 = 0), the
    total charge should be 0 (charge neutrality).
    """
    total = 0.0
    for n in range(N):
        baseline = 1.0 if n % 2 == 1 else 0.0
        s = "I" * n + "Z" + "I" * (N - 1 - n)
        Z_n = _pauli_string_matrix(s)
        exp_Z = float(np.real(psi.conj() @ Z_n @ psi))
        n_avg = 0.5 * (1 - exp_Z)
        total += n_avg - baseline
    return total


# ----------------------------------------------------------------------------
# Confinement / string tension demo
# ----------------------------------------------------------------------------

def string_tension(N: int, **kwargs) -> dict:
    """Estimate the string tension by comparing the vacuum energy with
    and without a small background electric field ε₀ = 0.5.

    For confining theories, finite ε₀ raises the energy linearly:
        E(ε₀) - E(0) ≈ (g² / 2) · (N - 1) · ε₀²
    if the ground state remains "in" the same charge sector.

    Returns:
        dict with E_0, E_eps, predicted_difference, observed_difference.
    """
    kwargs.pop("eps0", None)
    E_0, _ = schwinger_ground_state(N, eps0=0.0, **kwargs)
    E_eps, _ = schwinger_ground_state(N, eps0=0.5, **kwargs)
    g = kwargs.get("g", 1.0)
    predicted = 0.5 * g ** 2 * (N - 1) * (0.5) ** 2
    return {
        "E_0":                E_0,
        "E_eps":              E_eps,
        "observed_diff":      E_eps - E_0,
        "predicted_diff":     predicted,
        "N":                  N,
    }
