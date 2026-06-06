"""Pauli decomposition: write any 2^n × 2^n matrix as a sum of Pauli strings.

Any complex 2^n × 2^n matrix M can be uniquely expanded as:

    M  =  sum_P  c_P · P

where the sum runs over all 4^n Pauli strings P ∈ {I, X, Y, Z}^⊗n and

    c_P  =  Tr(P · M) / 2^n.

If M is Hermitian, all c_P are real. If M is unitary, the coefficients
satisfy sum |c_P|² = 1 (after normalization).

This module provides:

  - `decompose(M)`: return a list of (coef, pauli_string) pairs.
  - `reconstruct(terms)`: rebuild the matrix from the decomposition.
  - `pauli_weight_distribution(M)`: histogram of weights (#non-I letters)
    of the surviving Pauli strings — a measure of "locality" of M.
  - `is_diagonal_pauli_string(s)`: whether s only contains I and Z
    (relevant for measurement grouping).
  - `commuting_pauli_groups(strings)`: partition a list of Pauli strings
    into qubit-wise-commuting subsets (for efficient simultaneous
    measurement in VQE).

For n ≤ 6 this is tractable (4^n ≤ 4096 strings). Beyond that the
naive enumeration becomes too expensive — but the sparse output
(few large c_P's) is what matters for chemistry and physics
Hamiltonians.
"""

from __future__ import annotations

from itertools import product

import numpy as np


_PAULI = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


def pauli_string_matrix(s: str) -> np.ndarray:
    """Build the 2^n × 2^n matrix for a Pauli string of length n."""
    M = np.array([[1.0 + 0j]])
    for ch in s:
        M = np.kron(M, _PAULI[ch])
    return M


# ----------------------------------------------------------------------------
# Decomposition
# ----------------------------------------------------------------------------

def decompose(M: np.ndarray, tol: float = 1e-12
                ) -> list[tuple[complex, str]]:
    """Decompose a 2^n × 2^n matrix into Pauli strings.

    Returns:
        list of (coefficient, pauli_string) for nonzero terms only.
    """
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be a square matrix")
    d = M.shape[0]
    n = int(np.log2(d))
    if 2 ** n != d:
        raise ValueError(f"matrix size {d} is not a power of 2")

    terms: list[tuple[complex, str]] = []
    for combo in product("IXYZ", repeat=n):
        s = "".join(combo)
        P = pauli_string_matrix(s)
        coef = np.trace(P @ M) / d
        if abs(coef) > tol:
            terms.append((complex(coef), s))
    return terms


def reconstruct(terms: list[tuple[complex, str]]) -> np.ndarray:
    """Rebuild the matrix from its Pauli decomposition."""
    if not terms:
        raise ValueError("need at least one term")
    n = len(terms[0][1])
    d = 2 ** n
    M = np.zeros((d, d), dtype=np.complex128)
    for coef, s in terms:
        M += coef * pauli_string_matrix(s)
    return M


def decomposition_error(M: np.ndarray) -> float:
    """Round-trip a matrix through Pauli decomposition and back; return
    the Frobenius norm of the residual."""
    terms = decompose(M, tol=0)   # keep all 4^n terms
    M_rec = reconstruct(terms)
    return float(np.linalg.norm(M - M_rec))


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------

def pauli_weight(s: str) -> int:
    """Number of non-identity characters in a Pauli string."""
    return sum(1 for ch in s if ch != "I")


def pauli_weight_distribution(M: np.ndarray, tol: float = 1e-12
                                ) -> dict[int, int]:
    """For each weight k, count how many Pauli terms of that weight
    have |coef| > tol. Useful for assessing the "locality" of a
    Hamiltonian."""
    terms = decompose(M, tol=tol)
    dist: dict[int, int] = {}
    for _, s in terms:
        w = pauli_weight(s)
        dist[w] = dist.get(w, 0) + 1
    return dist


def is_diagonal_pauli_string(s: str) -> bool:
    """A Pauli string is diagonal in the computational basis iff it
    consists only of I and Z characters."""
    return all(ch in "IZ" for ch in s)


# ----------------------------------------------------------------------------
# Qubit-wise commuting groups
# ----------------------------------------------------------------------------

def qubit_wise_commute(a: str, b: str) -> bool:
    """Two Pauli strings are "qubit-wise commuting" (QWC) iff on EACH
    qubit independently, one of them is I or both are the same Pauli.

    This is a stronger condition than global commutation but gives the
    practical guarantee that the two Paulis can be measured by the SAME
    single-qubit measurement basis on each qubit.
    """
    for ca, cb in zip(a, b):
        if ca == "I" or cb == "I":
            continue
        if ca != cb:
            return False
    return True


def commuting_pauli_groups(strings: list[str]) -> list[list[str]]:
    """Greedy partition of Pauli strings into QWC groups.

    Algorithm: pick the first ungrouped string; create a group with it;
    add any other string that QWC-commutes with EVERY string currently
    in the group; repeat.

    Returns a list of groups (each a list of strings).
    """
    remaining = list(strings)
    groups: list[list[str]] = []
    while remaining:
        seed = remaining.pop(0)
        group = [seed]
        new_remaining = []
        for s in remaining:
            if all(qubit_wise_commute(s, g) for g in group):
                group.append(s)
            else:
                new_remaining.append(s)
        remaining = new_remaining
        groups.append(group)
    return groups


def measurement_basis_for_group(group: list[str]) -> str:
    """For a QWC group of strings, determine the measurement basis on
    each qubit.

    On each qubit, the basis is whichever non-I letter appears in some
    string of the group (X, Y, or Z); if all strings have I there,
    choose Z by convention.
    """
    if not group:
        return ""
    n = len(group[0])
    basis = []
    for q in range(n):
        non_i = set(s[q] for s in group if s[q] != "I")
        if not non_i:
            basis.append("Z")
        else:
            basis.append(non_i.pop())
    return "".join(basis)


# ----------------------------------------------------------------------------
# Shot-cost helpers
# ----------------------------------------------------------------------------

def shot_cost_naive(coefs: list[float], shots_per_term: int = 1000) -> int:
    """Naive: one separate measurement per term."""
    return shots_per_term * len(coefs)


def shot_cost_grouped(coefs: list[float], strings: list[str],
                        shots_per_group: int = 1000) -> int:
    """With QWC grouping, only `n_groups` separate measurement bases needed."""
    groups = commuting_pauli_groups(strings)
    return shots_per_group * len(groups)
