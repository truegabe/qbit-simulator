"""Mermin-GHZ inequality test.

Demonstrates that an N-qubit GHZ state violates Mermin's generalization of
the Bell-CHSH inequality by an exponentially growing factor.

Mermin's polynomial M_N has the property:
    Quantum (GHZ):           |⟨M_N⟩| = 2^(N-1)
    Classical (LHV bound):   |⟨M_N⟩| ≤ 2^(N/2)     (even N)
                                       2^((N-1)/2) (odd N)

So the quantum/classical ratio grows as 2^(N/2 - 1) or 2^((N-1)/2 - 1) — an
unbounded violation of any local-hidden-variable theory.

Cleanest formulation (Mermin 1990) for odd N:
    M_N = sum over Pauli strings on N qubits consisting of X's and Y's,
          with an EVEN number of Y's, signed by (-1)^(k/2) where k = #Y.

For N=3:    M_3 = XXX - XYY - YXY - YYX
For N=5:    M_5 = XXXXX
                 - (10 distinct strings with 2 Y's, signed -)
                 + (5 distinct strings with 4 Y's, signed +)
            etc.

Both quantum and classical bounds are computed exactly. This is the
canonical demonstration of multipartite quantum nonlocality.
"""

from __future__ import annotations

from itertools import combinations

from ..stabilizer import StabilizerState


def mermin_polynomial_terms(n: int) -> list[tuple[str, int]]:
    """Generate (pauli_string, sign) pairs for the Mermin polynomial M_N.

    Includes only Pauli strings with an even number of Y's, signed
    (-1)^(k/2) where k = number of Y's.
    """
    if n < 2:
        raise ValueError("Mermin polynomial needs N >= 2")
    terms: list[tuple[str, int]] = []
    for k in range(0, n + 1, 2):       # number of Y's, even only
        sign = (-1) ** (k // 2)
        for y_positions in combinations(range(n), k):
            s = ["X"] * n
            for p in y_positions:
                s[p] = "Y"
            terms.append(("".join(s), sign))
    return terms


def make_ghz(n: int) -> StabilizerState:
    """Build an N-qubit GHZ stabilizer state via H + CNOT cascade."""
    st = StabilizerState(n).h(0)
    for q in range(n - 1):
        st.cnot(q, q + 1)
    return st


def mermin_quantum_value(n: int) -> int:
    """Exact ⟨M_N⟩ on the N-qubit GHZ state."""
    state = make_ghz(n)
    total = 0
    for pauli, sign in mermin_polynomial_terms(n):
        total += sign * state.pauli_expectation(pauli)
    return total


def mermin_classical_bound(n: int) -> int:
    """Maximum |⟨M_N⟩| under any local hidden-variable theory.

    For a sum of products of ±1 variables, the LHV maximum is the L1 norm of
    the coefficient vector restricted to a particular factorization. For the
    Mermin polynomial with signs:
        even N: 2^(N/2)
        odd  N: 2^((N-1)/2)
    (See Mermin 1990; Werner & Wolf 2001 generalize.)
    """
    return 1 << (n // 2)        # 2^(N/2) for even N, 2^((N-1)/2) for odd N


def mermin_violation_report(n: int) -> dict:
    """Compute the Mermin test result for N qubits.

    Returns a dict with keys:
        n              : number of qubits
        terms          : how many Pauli strings appear in M_N
        quantum        : |⟨M_N⟩| on GHZ
        classical_bound: maximum |⟨M_N⟩| under local hidden variables
        violation      : quantum / classical_bound (the "Bell violation")
    """
    q = mermin_quantum_value(n)
    c = mermin_classical_bound(n)
    return {
        "n": n,
        "terms": len(mermin_polynomial_terms(n)),
        "quantum": abs(q),
        "classical_bound": c,
        "violation": abs(q) / c if c else float("inf"),
    }
