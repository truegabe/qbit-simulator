"""Grover's search algorithm — sparse N-qubit implementation.

The oracle is "flip the sign of one amplitude" (one element write, O(1)).
The diffuser `2|s><s| - I` over the uniform superposition `|s>` reduces to
`state -> 2*mean(state) - state` (O(2^N) time, no matrix built).

Memory: O(2^N) for the state vector. No 2^N x 2^N operators.
"""

from __future__ import annotations

import numpy as np

from ..circuit import QuantumCircuit


def optimal_iterations(n: int) -> int:
    N = 2**n
    return max(1, int(np.floor(np.pi / 4 * np.sqrt(N))))


def grover(n_qubits: int, marked: int, iterations: int | None = None) -> QuantumCircuit:
    if not 0 <= marked < 2**n_qubits:
        raise ValueError(f"marked must be in 0..{2**n_qubits - 1}")
    if iterations is None:
        iterations = optimal_iterations(n_qubits)

    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.h(q)

    for _ in range(iterations):
        # Oracle: phase-flip the marked basis state.
        qc.state[marked] = -qc.state[marked]
        # Diffuser: reflect about the mean amplitude.
        mean = qc.state.mean()
        qc.state = 2 * mean - qc.state
        qc.history.append(f"GroverStep(marked={marked})")

    return qc


def grover_2q(marked: int) -> QuantumCircuit:
    return grover(2, marked, iterations=1)
