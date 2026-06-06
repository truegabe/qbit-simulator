"""Quantum Amplitude Estimation (QAE) — Brassard, Hoyer, Mosca, Tapp 2002.

Generalization of Grover's algorithm: given a state-preparation unitary `A`
that produces

    |ψ⟩ = A|0...0⟩ = √(1-a) |bad⟩|0⟩  +  √a |good⟩|1⟩

(where the last qubit is the "good/bad" flag), QAE estimates the
amplitude `a ∈ [0, 1]` quadratically faster than classical sampling:
    classical:   ~1/ε² samples for precision ε
    quantum:     ~1/ε    queries to A for precision ε

How it works (one paragraph):

The Grover operator Q = -A·S_0·A^{-1}·S_ψ rotates |ψ⟩ inside a 2D subspace
by an angle 2θ, where sin²(θ) = a. Q has two eigenvalues e^{±2iθ}. Running
QPE on Q with input |ψ⟩ extracts θ from the counting register. The output
c ∈ [0, 2^t) gives an estimate φ ≈ c / 2^t of either θ/π or (1 - θ/π);
either way, a = sin²(πφ).

This is the canonical "QPE-based" QAE. There are newer iterative variants
(IQAE, MLQAE) that avoid the counting register at the cost of more shots;
those are not implemented here.

Applications (where this matters):
  - Quantum Monte Carlo: estimating expected values of random processes
  - Quantum finance: option pricing (Rebentrost-Lloyd, Stamatopoulos et al.)
  - Quantum machine learning: kernel evaluation, quantum k-means
  - Quantum chemistry: computing matrix elements faster than VQE
"""

from __future__ import annotations

import numpy as np

from ..circuit import QuantumCircuit
from .qpe import phase_estimation


def grover_operator(A: np.ndarray) -> np.ndarray:
    """Build the Grover-style operator Q for amplitude estimation.

    Q = -A · S_0 · A^{-1} · S_ψ
    where S_0 = I - 2|0...0⟩⟨0...0| (phase flip on the all-zeros state)
    and   S_ψ = I - 2 (I ⊗ |1⟩⟨1|) (phase flip on the flag qubit = |1⟩).

    Args:
        A: 2^(n+1) × 2^(n+1) state-prep unitary, last qubit is the flag.
    Returns:
        Q: same shape as A.
    """
    dim = A.shape[0]
    if A.shape != (dim, dim):
        raise ValueError("A must be square")
    if dim & (dim - 1):
        raise ValueError("A's dimension must be a power of 2")

    # S_ψ: phase flip whenever the last qubit (flag) is 1.
    # The last qubit's bit in an index `i` is `i & 1`.
    S_psi = np.eye(dim, dtype=np.complex128)
    for i in range(dim):
        if i & 1:
            S_psi[i, i] = -1.0

    # S_0: phase flip on the all-zeros state.
    S_0 = np.eye(dim, dtype=np.complex128)
    S_0[0, 0] = -1.0

    A_dag = A.conj().T
    return -A @ S_0 @ A_dag @ S_psi


def amplitude_estimation(
    A: np.ndarray,
    n_counting: int = 8,
) -> dict:
    """Estimate the amplitude `a` of the good state prepared by A.

    Args:
        A: state-preparation unitary on (n+1) qubits; A|0...0⟩ has the form
           √(1-a) |bad⟩|0⟩ + √a |good⟩|1⟩.
        n_counting: number of counting qubits (precision = π / 2^n_counting).

    Returns:
        dict with:
          - amplitude:           best estimate of a
          - theta_estimate:      best estimate of θ (where sin²(θ) = a)
          - phase:               raw QPE phase output in [0, 1)
          - counting_marginal:   probability distribution over counting outcomes
          - top_two_estimates:   the two most likely a-values (Q has ±2θ pair)
          - n_qubits:            n + 1 + n_counting
    """
    dim = A.shape[0]
    if dim & (dim - 1):
        raise ValueError("A's dimension must be a power of 2")
    n_work = int(np.log2(dim))   # n + 1 = work qubits including the flag

    Q = grover_operator(A)

    # The state |ψ⟩ = A|0...0⟩ sits in the 2D subspace spanned by the two
    # eigenvectors of Q with eigenvalues e^{±2iθ}. QPE on this state will
    # return one of φ = ±θ/π (mod 1), each with probability ~1/2.
    psi = A[:, 0]                          # A|0⟩ is the first column

    qc = phase_estimation(Q, psi, n_counting)

    # Marginal over the counting register.
    probs = qc.probabilities()
    marginal = probs.reshape(1 << n_counting, 1 << n_work).sum(axis=1)

    # Top two outcomes (symmetric around 0.5 if θ ∈ (0, π/2)).
    top_two_idx = np.argsort(marginal)[-2:][::-1]

    def _phase_to_amplitude(c: int) -> tuple[float, float, float]:
        phi = c / (1 << n_counting)
        # θ = π · min(φ, 1-φ) gives the principal value
        theta = np.pi * min(phi, 1.0 - phi)
        return phi, theta, float(np.sin(theta) ** 2)

    # Best estimate
    best_c = int(np.argmax(marginal))
    phi, theta, amp = _phase_to_amplitude(best_c)

    top_two_estimates = [_phase_to_amplitude(int(c))[2] for c in top_two_idx]

    return {
        "amplitude": amp,
        "theta_estimate": theta,
        "phase": phi,
        "counting_marginal": marginal,
        "top_two_estimates": top_two_estimates,
        "n_qubits": n_work + n_counting,
    }


def make_ry_test_unitary(theta: float) -> np.ndarray:
    """Build a 1-qubit Ry(2θ) -- a convenient test state-prep with a = sin²(θ)."""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s,  c]], dtype=np.complex128)
