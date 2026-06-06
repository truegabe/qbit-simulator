"""Iterative Quantum Phase Estimation (IQPE) — Kitaev 1995.

Estimates the phase φ of U|ψ⟩ = e^{2πiφ}|ψ⟩ using only ONE ancilla qubit,
instead of the t counting qubits standard QPE needs. The trade-off: t
sequential measurements (each conditioned on earlier outcomes) instead of
a single batched read of t qubits.

How it works (bit-by-bit, least to most significant):

For k = t-1, t-2, ..., 0:
    1. Set ancilla to |+⟩.
    2. Apply controlled-U^(2^k) using the ancilla as control.
    3. Apply a Z-rotation of angle ω_k = -2π · Σ_{j=k+1}^{t-1} φ_j / 2^(j-k+1)
       to the ancilla. This feedback erases the "earlier" bits we've already
       measured, leaving only the k-th bit of φ on the ancilla.
    4. Apply H to ancilla, measure it. The outcome is φ_k (the k-th bit of
       the binary expansion of φ).

After t iterations, φ ≈ Σ_k φ_k / 2^(k+1).

Advantages over standard QPE:
    - Uses 1 + dim(ψ) qubits instead of t + dim(ψ).
    - Each measurement happens after only depth-O(t) of U^(2^k); good for
      NISQ devices with limited coherence.
"""

from __future__ import annotations

import numpy as np

from ..circuit import QuantumCircuit


def iterative_qpe(
    U: np.ndarray,
    eigenstate: np.ndarray,
    n_bits: int,
    rng: np.random.Generator | None = None,
) -> dict:
    """Run iterative QPE.

    Args:
        U:          d × d unitary (d = 2^k for some k).
        eigenstate: length-d eigenstate of U.
        n_bits:     precision (number of phase bits to extract).
        rng:        numpy generator for ancilla measurements.

    Returns:
        dict with:
            phase:       estimated φ ∈ [0, 1)
            bits:        list of measured phase bits, MSB first
            measurements: sequence of single-shot measurement outcomes
    """
    rng = rng or np.random.default_rng()
    d = U.shape[0]
    n_eig = int(np.log2(d))
    if 2**n_eig != d:
        raise ValueError("U's dimension must be a power of 2")

    eigenstate = np.asarray(eigenstate, dtype=np.complex128)
    eigenstate /= np.linalg.norm(eigenstate)

    # Pre-compute powers of U: U^(2^0), U^(2^1), ..., U^(2^(n_bits-1)).
    U_powers = [U.copy()]
    for _ in range(1, n_bits):
        U_powers.append(U_powers[-1] @ U_powers[-1])

    phi_bits: list[int] = [0] * n_bits   # bits[k] is the k-th bit of φ

    # Iterate from least significant to most significant bit.
    for k in range(n_bits - 1, -1, -1):
        # Feedback angle from already-measured later bits.
        omega = 0.0
        for j in range(k + 1, n_bits):
            omega -= 2 * np.pi * phi_bits[j] / (2 ** (j - k + 1))

        # Set up a tiny circuit: 1 ancilla + n_eig eigenstate qubits.
        qc = QuantumCircuit(1 + n_eig)
        # Load eigenstate into the lower (n_eig) qubits. The first qubit
        # (ancilla) starts in |0⟩.
        full = np.zeros(2 * d, dtype=np.complex128)
        for j, amp in enumerate(eigenstate):
            full[j] = amp           # ancilla = 0 block
        qc.state = full

        # Hadamard on ancilla.
        qc.h(0)

        # Controlled-U^(2^k) with control = ancilla. Build the matrix
        # explicitly as a 2d × 2d block-diagonal: I on |0⟩_a block, U^(2^k) on
        # |1⟩_a block.
        CU = np.eye(2 * d, dtype=np.complex128)
        CU[d:, d:] = U_powers[k]
        qc.apply_unitary(CU, [0] + list(range(1, 1 + n_eig)), check_unitary=False)

        # Phase feedback: Rz on ancilla.
        qc.p(omega, 0)

        # Hadamard then measure.
        qc.h(0)
        # Measure qubit 0 (the ancilla).
        probs = qc.probabilities().reshape(2, d).sum(axis=1)
        probs = np.clip(probs, 0, None)
        probs /= probs.sum()
        outcome = int(rng.choice(2, p=probs))
        phi_bits[k] = outcome

    # Reconstruct φ = Σ_k phi_bits[k] / 2^(k+1).
    phi = sum(phi_bits[k] / (2 ** (k + 1)) for k in range(n_bits))

    return {
        "phase":         phi,
        "bits":          phi_bits[::-1],     # MSB first for display
        "n_bits":        n_bits,
    }
