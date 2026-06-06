"""Fermion operators + Jordan-Wigner mapping.

A FermionOp represents a linear combination of products of fermionic
creation (c†) and annihilation (c) operators. The `to_pauli_op` method
applies the Jordan-Wigner transform:

    c_j   = (Z_0 Z_1 ... Z_{j-1}) (X_j + i Y_j) / 2
    c†_j  = (Z_0 Z_1 ... Z_{j-1}) (X_j - i Y_j) / 2
    n_j   = c†_j c_j = (I - Z_j) / 2

This bridges the existing PauliOp + VQE infrastructure to second-
quantized Hamiltonians of interest in condensed matter and chemistry.

Provided here:
    - FermionOp class with c(), cdag(), number()
    - Arithmetic: addition, scalar multiplication, fermion-op multiplication
    - to_pauli_op(n_modes) -> PauliOp via Jordan-Wigner
    - hubbard_hamiltonian(L, t, U) -> 2L-mode FermionOp for the 1D Hubbard model
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .pauli import PauliOp


# ----------------------------------------------------------------------------
# Pauli string algebra (used by Jordan-Wigner)
# ----------------------------------------------------------------------------

_PAULI_MUL = {
    "II": (1 + 0j, "I"),  "IX": (1 + 0j, "X"),  "IY": (1 + 0j, "Y"),  "IZ": (1 + 0j, "Z"),
    "XI": (1 + 0j, "X"),  "XX": (1 + 0j, "I"),  "XY": (1j,     "Z"),  "XZ": (-1j,    "Y"),
    "YI": (1 + 0j, "Y"),  "YX": (-1j,    "Z"),  "YY": (1 + 0j, "I"),  "YZ": (1j,     "X"),
    "ZI": (1 + 0j, "Z"),  "ZX": (1j,     "Y"),  "ZY": (-1j,    "X"),  "ZZ": (1 + 0j, "I"),
}


def _pauli_string_mul(a: str, b: str) -> tuple[complex, str]:
    """Multiply two Pauli strings of equal length. Returns (coefficient, result)."""
    if len(a) != len(b):
        raise ValueError("Pauli strings must have equal length")
    coef = 1 + 0j
    out_chars = []
    for ca, cb in zip(a, b):
        c, p = _PAULI_MUL[ca + cb]
        coef *= c
        out_chars.append(p)
    return coef, "".join(out_chars)


def _pauli_terms_simplify(terms: list[tuple[complex, str]]
                          ) -> list[tuple[complex, str]]:
    """Sum like terms; drop terms with negligible coefficient."""
    bucket: dict[str, complex] = {}
    for coef, s in terms:
        bucket[s] = bucket.get(s, 0 + 0j) + coef
    return [(c, s) for s, c in bucket.items() if abs(c) > 1e-12]


# ----------------------------------------------------------------------------
# FermionOp
# ----------------------------------------------------------------------------

# A "term" in a FermionOp is a tuple (coefficient, ops) where `ops` is a tuple
# of (mode_index: int, is_dagger: bool) pairs, ordered left-to-right.
FermionTerm = tuple[complex, tuple[tuple[int, bool], ...]]


class FermionOp:
    """Linear combination of products of fermionic creation/annihilation ops.

    Internal representation: list of (coefficient, op_tuple) where op_tuple is
    a tuple of (mode, is_dagger). No normal ordering is performed; the
    Jordan-Wigner transform handles ordering implicitly when projecting to
    Pauli strings.
    """

    def __init__(self, terms: Iterable[FermionTerm] | None = None):
        self.terms: list[FermionTerm] = list(terms) if terms else []

    # ---- constructors ----

    @classmethod
    def zero(cls) -> "FermionOp":
        return cls()

    @classmethod
    def identity(cls) -> "FermionOp":
        return cls([(1 + 0j, ())])

    @classmethod
    def c(cls, mode: int) -> "FermionOp":
        """Annihilation operator c_mode."""
        if mode < 0:
            raise ValueError("mode must be non-negative")
        return cls([(1 + 0j, ((mode, False),))])

    @classmethod
    def cdag(cls, mode: int) -> "FermionOp":
        """Creation operator c†_mode."""
        if mode < 0:
            raise ValueError("mode must be non-negative")
        return cls([(1 + 0j, ((mode, True),))])

    @classmethod
    def number(cls, mode: int) -> "FermionOp":
        """Number operator n_mode = c†_mode c_mode."""
        return cls.cdag(mode) * cls.c(mode)

    # ---- arithmetic ----

    def __add__(self, other: "FermionOp") -> "FermionOp":
        if not isinstance(other, FermionOp):
            raise TypeError("can only add FermionOp")
        return FermionOp(self.terms + other.terms)

    def __sub__(self, other: "FermionOp") -> "FermionOp":
        return self + (-1.0) * other

    def __rmul__(self, scalar: complex) -> "FermionOp":
        return FermionOp([(scalar * c, ops) for c, ops in self.terms])

    def __neg__(self) -> "FermionOp":
        return (-1.0) * self

    def __mul__(self, other: "FermionOp") -> "FermionOp":
        if not isinstance(other, FermionOp):
            raise TypeError("can only multiply by another FermionOp or scalar")
        new_terms: list[FermionTerm] = []
        for c1, ops1 in self.terms:
            for c2, ops2 in other.terms:
                new_terms.append((c1 * c2, ops1 + ops2))
        return FermionOp(new_terms)

    # ---- inspection ----

    def n_modes(self) -> int:
        """Smallest n such that every mode index used is < n."""
        if not self.terms:
            return 0
        return max((op[0] for _, ops in self.terms for op in ops), default=-1) + 1

    def __repr__(self) -> str:
        if not self.terms:
            return "FermionOp(0)"
        parts = []
        for c, ops in self.terms:
            op_strs = " ".join(f"c{'†' if d else ''}_{m}" for m, d in ops)
            parts.append(f"({c:+.4g}) {op_strs}".rstrip())
        return "FermionOp[" + " + ".join(parts) + "]"

    # ---- Jordan-Wigner ----

    def to_pauli_op(self, n_modes: int | None = None) -> PauliOp:
        """Apply the Jordan-Wigner transform to produce a PauliOp."""
        n = n_modes if n_modes is not None else self.n_modes()
        if n == 0:
            # Constant operator (just the identity term).
            const = sum((c for c, ops in self.terms if not ops), 0j)
            return PauliOp([(const, "I")])

        all_pauli_terms: list[tuple[complex, str]] = []
        for coef, ops in self.terms:
            # Build the Pauli expansion of this fermionic monomial.
            current = [(complex(coef), "I" * n)]
            for (mode, is_dagger) in ops:
                next_terms: list[tuple[complex, str]] = []
                jw_terms = _jw_fermion_op(mode, is_dagger, n)
                for c_a, s_a in current:
                    for c_b, s_b in jw_terms:
                        c, s = _pauli_string_mul(s_a, s_b)
                        next_terms.append((c_a * c_b * c, s))
                current = _pauli_terms_simplify(next_terms)
            all_pauli_terms.extend(current)

        all_pauli_terms = _pauli_terms_simplify(all_pauli_terms)
        if not all_pauli_terms:
            # Operator simplifies to zero; return a tiny identity-coefficient.
            return PauliOp([(0 + 0j, "I" * n)])
        return PauliOp(all_pauli_terms)


def _jw_fermion_op(mode: int, is_dagger: bool, n: int) -> list[tuple[complex, str]]:
    """Jordan-Wigner image of a single c_mode or c†_mode on `n` modes.

    Returns a 2-term list:
        c_j   = (Z⊗...⊗Z⊗(X + i Y)/2 ⊗ I⊗...⊗I)
        c†_j  = (Z⊗...⊗Z⊗(X - i Y)/2 ⊗ I⊗...⊗I)
    The Z-string runs over modes 0..j-1.
    """
    if mode >= n:
        raise IndexError(f"mode {mode} >= n_modes {n}")
    base = ["Z"] * mode + [""] + ["I"] * (n - mode - 1)
    # Two contributing Pauli strings: one with X at `mode`, one with Y.
    s_x = base.copy(); s_x[mode] = "X"
    s_y = base.copy(); s_y[mode] = "Y"
    if is_dagger:
        return [(0.5 + 0j, "".join(s_x)), (-0.5j, "".join(s_y))]
    else:
        return [(0.5 + 0j, "".join(s_x)), (0.5j,  "".join(s_y))]


# ----------------------------------------------------------------------------
# The Fermi-Hubbard model (canonical benchmark for fermionic simulation)
# ----------------------------------------------------------------------------

def total_particle_number(n_modes: int) -> FermionOp:
    """Total particle number operator: N = Σ_i c†_i c_i."""
    op = FermionOp.zero()
    for i in range(n_modes):
        op = op + FermionOp.number(i)
    return op


def project_to_particle_number_sector(
    H_matrix: np.ndarray, n_target: int, tol: float = 0.5,
) -> np.ndarray:
    """Restrict H to the subspace with `n_target` total particles.

    Diagonalize the total-N operator, pick eigenvectors with eigenvalue
    close to `n_target` (default tol=0.5 — integer N's are well-separated),
    and project H onto that subspace.

    Returns a smaller matrix of shape (binom(n_modes, n_target), ...).
    """
    n_modes = int(np.log2(H_matrix.shape[0]))
    if 2**n_modes != H_matrix.shape[0]:
        raise ValueError("H must have dimension 2^n for some integer n")
    N_op = total_particle_number(n_modes).to_pauli_op(n_modes).matrix()
    eigvals, eigvecs = np.linalg.eigh(N_op)
    mask = np.abs(eigvals - n_target) < tol
    V = eigvecs[:, mask]
    if V.shape[1] == 0:
        raise ValueError(f"no states with N = {n_target} in this Hilbert space")
    return V.conj().T @ H_matrix @ V


def hubbard_hamiltonian(
    L: int,
    t: float = 1.0,
    U: float = 4.0,
    periodic: bool = False,
) -> FermionOp:
    """1D Fermi-Hubbard model on L sites with on-site interaction U and hopping t.

    Spin-orbital indexing: mode 2k = site k spin-up, mode 2k+1 = site k spin-down.
    H = -t Σ_{<k,k+1>, σ} (c†_{k,σ} c_{k+1,σ} + h.c.)
        + U Σ_k n_{k,↑} n_{k,↓}

    Args:
        L: number of sites.
        t: hopping amplitude (default 1.0).
        U: on-site Coulomb interaction (default 4.0).
        periodic: if True, also include the wrap-around bond (k=L-1, k=0).

    Returns:
        FermionOp on 2L modes.
    """
    if L < 2:
        raise ValueError("Hubbard model needs at least 2 sites")

    def mode(site: int, spin: int) -> int:        # spin: 0=up, 1=down
        return 2 * site + spin

    H = FermionOp.zero()
    # Hopping terms.
    bonds = list(range(L - 1))
    if periodic:
        bonds.append(L - 1)              # last->first wrap
    for k in bonds:
        k_next = (k + 1) % L
        for spin in (0, 1):
            i = mode(k, spin)
            j = mode(k_next, spin)
            hop = FermionOp.cdag(i) * FermionOp.c(j) + FermionOp.cdag(j) * FermionOp.c(i)
            H = H + (-t) * hop
    # On-site Coulomb.
    for k in range(L):
        n_up = FermionOp.number(mode(k, 0))
        n_dn = FermionOp.number(mode(k, 1))
        H = H + U * (n_up * n_dn)
    return H
