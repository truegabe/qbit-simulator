"""Trotter-Suzuki decomposition for general Pauli-sum Hamiltonians.

Given H = Σ_k c_k P_k where P_k is a Pauli string, compute e^{-iHt}
approximately as a product of single-Pauli rotations:

    Trotter order 1 (Lie):     ∏_k exp(-i c_k P_k t) per step
    Trotter order 2 (Strang):  ∏_k exp(-i c_k P_k t/2) · ∏_{k rev} exp(-i c_k P_k t/2)

Each `exp(-iθP)` for a Pauli string P is a "Pauli rotation" — implementable
as O(weight(P)) CNOTs plus one Rz rotation:

    1. Basis-change each non-Z Pauli to Z (H for X, S†HS for Y).
    2. CNOT-cascade reduces to one effective qubit.
    3. Rz(2θ) on that qubit.
    4. Uncompute CNOT cascade and basis change.

This is the standard digital-quantum-simulation primitive (Lloyd 1996).
"""

from __future__ import annotations

import numpy as np

from ..circuit import QuantumCircuit
from ..pauli import PauliOp


def apply_pauli_rotation(qc: QuantumCircuit, pauli: str, theta: float) -> None:
    """Apply exp(-i θ P) where P is a Pauli string, in place on `qc`.

    Decomposition: basis-change to Z, CNOT cascade, Rz(2θ), uncompute.
    """
    if len(pauli) != qc.n:
        raise ValueError(f"Pauli {pauli!r} doesn't match {qc.n} qubits")
    # Skip the identity case.
    if all(c == "I" for c in pauli):
        # Global phase only — no-op for our state representation.
        return

    # 1. Basis change to Z on each non-I, non-Z qubit.
    #    For X:  apply H        (H·X·H = Z).
    #    For Y:  apply S†, then H  (H·S†·Y·S·H = Z).
    support = []
    for q, ch in enumerate(pauli):
        if ch == "I":
            continue
        support.append(q)
        if ch == "X":
            qc.h(q)
        elif ch == "Y":
            _apply_sdg(qc, q)
            qc.h(q)
    # 2. CNOT cascade: pair up support qubits, leaving one "root" at the end.
    if len(support) >= 2:
        for k in range(len(support) - 1):
            qc.cnot(support[k], support[-1])
    # 3. Rotation on the root qubit.
    root = support[-1]
    qc.rz(2 * theta, root)
    # 4. Uncompute CNOT cascade (reverse order).
    if len(support) >= 2:
        for k in reversed(range(len(support) - 1)):
            qc.cnot(support[k], support[-1])
    # 5. Uncompute basis change (inverse of step 1, applied in reverse order
    #    within each qubit, and reverse-order across qubits doesn't matter here
    #    since each qubit's gates only act on itself).
    for q in support:
        ch = pauli[q]
        if ch == "X":
            qc.h(q)
        elif ch == "Y":
            # Inverse of (S†; H) is (H; S).
            qc.h(q)
            qc.s(q)


def _apply_sdg(qc: QuantumCircuit, q: int) -> None:
    """S-dagger via three S gates (S† = S³ since S has order 4)."""
    if hasattr(qc, "sdg"):
        qc.sdg(q)
    else:
        qc.s(q); qc.s(q); qc.s(q)


def trotter_step(
    qc: QuantumCircuit, H: PauliOp, dt: float, order: int = 2,
) -> None:
    """Apply one Trotter-Suzuki step exp(-i H dt) in place.

    For Hermitian PauliOp H = Σ c_k P_k with real c_k, each term contributes
    exp(-i c_k P_k dt) = apply_pauli_rotation(P_k, c_k · dt).
    """
    terms = list(H.terms)
    if order == 1:
        for c, P in terms:
            if abs(c.imag) > 1e-9:
                raise ValueError("Trotter assumes Hermitian H (real coeffs)")
            apply_pauli_rotation(qc, P, float(c.real) * dt)
    elif order == 2:
        # Strang: first half forwards, then full middle... but for a sum of
        # ALL non-commuting terms we use the symmetric splitting:
        # ∏_k Pauli(θ_k/2) for k forwards, then ∏_k Pauli(θ_k/2) reversed.
        for c, P in terms:
            apply_pauli_rotation(qc, P, float(c.real) * dt / 2)
        for c, P in reversed(terms):
            apply_pauli_rotation(qc, P, float(c.real) * dt / 2)
    else:
        raise ValueError(f"only order 1 or 2 supported, got {order}")


def trotter_evolve(
    H: PauliOp,
    initial_state: np.ndarray,
    total_time: float,
    n_steps: int = 50,
    order: int = 2,
) -> np.ndarray:
    """Evolve `initial_state` under H for `total_time`, returning final state.

    Builds a circuit, not a single matrix exponentiation — useful for
    verifying digital simulation against the analytic answer.
    """
    n = int(np.log2(initial_state.size))
    qc = QuantumCircuit(n)
    qc.state = initial_state.astype(np.complex128).copy()
    dt = total_time / n_steps
    for _ in range(n_steps):
        trotter_step(qc, H, dt, order=order)
    return qc.state
