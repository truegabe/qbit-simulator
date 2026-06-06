"""Quantum counting (Brassard, Hoyer, Tapp 1998).

Given an oracle that marks M items in a search space of size N, estimate M
using quantum amplitude estimation. Applies QAE directly to the Grover
diffusion operator built from the oracle.

Classical solution: needs O(N) queries in the worst case (you have to scan).
Quantum (counting via QAE): O(√N / ε) queries for relative error ε.
That's the same quadratic advantage Grover gives for *finding* a marked
item, except here we *count* without finding any.

Use case: estimate the size of a solution set when you can write the
oracle but not the explicit indicator.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .amplitude_estimation import amplitude_estimation


def _oracle_unitary(n: int, marked: set[int]) -> np.ndarray:
    """Build the standard Grover oracle: phase-flip on marked basis states.

    Returns a 2^n × 2^n diagonal unitary with -1 on marked indices, +1 elsewhere.
    """
    N = 1 << n
    O = np.eye(N, dtype=np.complex128)
    for m in marked:
        if not (0 <= m < N):
            raise IndexError(f"marked index {m} out of range [0, {N})")
        O[m, m] = -1.0
    return O


def quantum_count(
    n: int,
    marked: set[int] | Callable[[int], bool],
    n_counting: int = 8,
) -> dict:
    """Estimate how many states are marked by the given oracle.

    Args:
        n:           number of qubits in the search register (search space N = 2^n).
        marked:      either a set of marked basis-state indices, OR a predicate
                     `f(idx) -> bool` that classifies each index.
        n_counting:  precision of the QAE phase estimation.

    Returns:
        dict with:
            M_estimate:   estimated number of marked items
            N:            total search space size (2^n)
            amplitude:    estimated sin²(θ) where M/N = sin²(θ)
            confidence:   ratio of the top two QAE peaks
    """
    N = 1 << n
    if callable(marked):
        marked_set = {i for i in range(N) if marked(i)}
    else:
        marked_set = set(marked)

    # State-prep unitary: a single H on every qubit (uniform superposition),
    # with an extra flag qubit set by the oracle's phase.
    # The trick: append a flag qubit that gets X'd on marked indices.
    #
    # Concretely we use A acting on (n+1) qubits where the first n are the
    # search register and the last is the flag. A:
    #   1. H on each search qubit (uniform superposition of N basis states).
    #   2. CNOT-style "phase oracle": flip the flag if search ∈ marked.
    # Then the amplitude of |1⟩_flag is √(M/N) = sin(θ), so a = M/N.

    dim_n = N
    dim_total = 2 * N
    # Step 1: Hadamards on search register, flag stays |0⟩.
    H_single = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
    H_n = np.array([[1.0]], dtype=np.complex128)
    for _ in range(n):
        H_n = np.kron(H_n, H_single)
    I_flag = np.eye(2, dtype=np.complex128)
    H_block = np.kron(H_n, I_flag)

    # Step 2: oracle as a permutation -- if search index is in `marked_set`,
    # flip the flag qubit (controlled-X with the marked indices as controls).
    # Build directly:
    oracle = np.eye(dim_total, dtype=np.complex128)
    for s in range(dim_n):
        if s in marked_set:
            # Swap rows/cols (2s) ↔ (2s + 1) — XOR the flag bit.
            i0 = 2 * s
            i1 = 2 * s + 1
            oracle[[i0, i1]] = oracle[[i1, i0]]

    A = oracle @ H_block

    # Run QAE on this A.
    qae_result = amplitude_estimation(A, n_counting=n_counting)
    a = qae_result["amplitude"]
    M_est = a * N

    # Confidence: ratio of dominant peak to second peak in counting marginal.
    marginal = qae_result["counting_marginal"]
    top = np.sort(marginal)[::-1]
    confidence = float(top[0] / (top[1] + 1e-12)) if len(top) > 1 else float("inf")

    return {
        "M_estimate":    M_est,
        "M_estimate_int": int(round(M_est)),
        "N":             N,
        "amplitude":     a,
        "confidence":    confidence,
        "n_counting":    n_counting,
        "actual_M":      len(marked_set),
    }
