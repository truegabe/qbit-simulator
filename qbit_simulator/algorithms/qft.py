"""Quantum Fourier Transform on N qubits.

Implements the textbook construction with H + controlled-phase gates, then
swaps to reverse qubit order so the output matches the standard DFT ordering.
"""

from __future__ import annotations

import numpy as np

from ..circuit import QuantumCircuit


def apply_qft(qc: QuantumCircuit, qubits: list[int] | None = None) -> QuantumCircuit:
    if qubits is None:
        qubits = list(range(qc.n))
    n = len(qubits)
    for j in range(n):
        qc.h(qubits[j])
        for k in range(j + 1, n):
            angle = np.pi / (2 ** (k - j))
            qc.cp(angle, qubits[k], qubits[j])
    # Reverse qubit order. When operating on all qubits this is one O(2^N)
    # transpose; otherwise fall back to pairwise SWAPs on the chosen subset.
    if qubits == list(range(qc.n)):
        qc.reverse_qubits()
    else:
        for j in range(n // 2):
            qc.swap(qubits[j], qubits[n - 1 - j])
    return qc


def qft(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    return apply_qft(qc)


def qft_matrix(n: int) -> np.ndarray:
    """Reference DFT matrix for verification."""
    N = 2**n
    omega = np.exp(2j * np.pi / N)
    j, k = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    return omega ** (j * k) / np.sqrt(N)
