"""QAOA — Quantum Approximate Optimization Algorithm — on Max-Cut.

For a graph G=(V,E), assign each vertex i a binary variable z_i ∈ {0,1}.
The cost of a cut is the number of edges with endpoints in different sets:
    C(z) = Σ_{(i,j) ∈ E} (1 - z_i z_j_sign) / 2   (using ±1 spins)

QAOA encodes C as a Hamiltonian:
    H_C = Σ_{(i,j) ∈ E} (I - Z_i Z_j) / 2

and applies p layers of e^{-iγ H_C} · e^{-iβ H_B} where H_B = Σ X_i,
starting from the uniform superposition. The 2p parameters {γ_k, β_k} are
optimized classically to maximize ⟨ψ|H_C|ψ⟩.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..circuit import QuantumCircuit
from ..pauli import PauliOp
from .vqe import nelder_mead


def maxcut_hamiltonian(edges: Sequence[tuple[int, int]], n_qubits: int) -> PauliOp:
    """Build the Max-Cut cost Hamiltonian for the given graph."""
    terms: list[tuple[complex, str]] = []
    # Sum of edge contributions: (I - Z_i Z_j) / 2.
    # We accumulate by adding individual Pauli strings; the PauliOp constructor
    # doesn't auto-combine like terms, but expectation values add up correctly.
    n_edges = len(edges)
    if n_edges:
        terms.append((complex(n_edges / 2), "I" * n_qubits))
    for (i, j) in edges:
        s = ["I"] * n_qubits
        s[i] = "Z"; s[j] = "Z"
        terms.append((complex(-0.5), "".join(s)))
    return PauliOp(terms)


def qaoa_ansatz(
    edges: Sequence[tuple[int, int]],
    n_qubits: int,
    gammas: Sequence[float],
    betas: Sequence[float],
) -> QuantumCircuit:
    """Build the QAOA ansatz with p layers (p = len(gammas) = len(betas))."""
    if len(gammas) != len(betas):
        raise ValueError("gammas and betas must have the same length")
    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.h(q)
    for gamma, beta in zip(gammas, betas):
        # e^{-i γ H_C}: each ZZ term factor exp(-i γ/2 · -Z_i Z_j) = exp(i γ/2 · Z_i Z_j)
        # Implementation: CNOT(i,j); Rz(-γ, j); CNOT(i,j)
        # Note: (I/2) contributions are global phase, ignored.
        for (i, j) in edges:
            qc.cnot(i, j)
            qc.rz(-gamma, j)
            qc.cnot(i, j)
        # e^{-i β H_B}: Rx(2β) on each qubit
        for q in range(n_qubits):
            qc.rx(2 * beta, q)
    return qc


def qaoa(
    edges: Sequence[tuple[int, int]],
    n_qubits: int,
    p: int = 1,
    seed: int | None = None,
) -> tuple[np.ndarray, float, list[float]]:
    """Run QAOA with p layers. Returns (optimal_params, max_cost, trace).

    optimal_params is a length-2p array: [γ_0..γ_{p-1}, β_0..β_{p-1}].
    """
    H = maxcut_hamiltonian(edges, n_qubits)
    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(0.0, np.pi, size=2 * p)
    trace: list[float] = []

    def neg_cost(theta: np.ndarray) -> float:
        gammas, betas = theta[:p], theta[p:]
        qc = qaoa_ansatz(edges, n_qubits, gammas, betas)
        c = H.expectation(qc.state)
        trace.append(c)
        return -c  # minimize negative = maximize cost

    theta_opt, neg_e_opt = nelder_mead(neg_cost, theta0, step=0.3, max_iter=500)
    return theta_opt, -neg_e_opt, trace


def sample_maxcut_solution(
    edges: Sequence[tuple[int, int]],
    n_qubits: int,
    theta_opt: np.ndarray,
    shots: int = 1024,
    seed: int | None = None,
) -> dict[str, int]:
    """Run the optimized circuit and return measurement counts."""
    p = len(theta_opt) // 2
    gammas, betas = theta_opt[:p], theta_opt[p:]
    qc = qaoa_ansatz(edges, n_qubits, gammas, betas)
    rng = np.random.default_rng(seed)
    return qc.counts(shots=shots, rng=rng)
