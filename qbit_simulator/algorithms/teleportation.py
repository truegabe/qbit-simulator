"""Quantum teleportation protocol.

Alice has an arbitrary qubit state |ψ⟩ = α|0⟩ + β|1⟩ on qubit 0.
Alice and Bob share a Bell pair across qubits 1 (Alice) and 2 (Bob).
Alice performs a Bell-basis measurement on qubits 0 and 1, gets two
classical bits (m0, m1), and sends them to Bob.
Bob applies X^m1 then Z^m0 to qubit 2, recovering |ψ⟩.

Key property: the original state on qubit 0 is destroyed by measurement
(no-cloning theorem holds). Only one perfect copy ever exists.
"""

from __future__ import annotations

import numpy as np

from ..circuit import QuantumCircuit


def teleport_state(
    alpha: complex,
    beta: complex,
    rng: np.random.Generator | None = None,
) -> tuple[QuantumCircuit, complex, complex, tuple[int, int]]:
    """Teleport the single-qubit state α|0⟩ + β|1⟩ from qubit 0 to qubit 2.

    Returns (circuit, alpha_received, beta_received, (m0, m1)).
    """
    rng = rng or np.random.default_rng()
    # Normalize input state.
    norm = np.sqrt(abs(alpha) ** 2 + abs(beta) ** 2)
    alpha = complex(alpha) / norm
    beta = complex(beta) / norm

    qc = QuantumCircuit(3)
    # Step 1: prepare arbitrary state on qubit 0 by directly setting amplitudes.
    qc.state[:] = 0
    qc.state[0b000] = alpha
    qc.state[0b100] = beta  # q0 = 1, q1 = 0, q2 = 0

    # Step 2: create Bell pair on qubits 1, 2.
    qc.h(1)
    qc.cnot(1, 2)

    # Step 3: Bell measurement on qubits 0, 1.
    qc.cnot(0, 1)
    qc.h(0)
    m0 = qc.measure_qubit(0, rng=rng)
    m1 = qc.measure_qubit(1, rng=rng)

    # Step 4: classically-conditioned correction on qubit 2.
    if m1 == 1:
        qc.x(2)
    if m0 == 1:
        qc.z(2)

    # Extract the post-protocol state of qubit 2.
    # After measurement and corrections, qubits 0 and 1 are in computational basis
    # state (m0, m1); the amplitudes for qubit 2 live at the corresponding indices.
    base = (m0 << 2) | (m1 << 1)
    alpha_recv = qc.state[base | 0]
    beta_recv = qc.state[base | 1]
    return qc, alpha_recv, beta_recv, (m0, m1)


def fidelity(a1: complex, b1: complex, a2: complex, b2: complex) -> float:
    """|⟨ψ₁|ψ₂⟩|² between two single-qubit states (each (α, β))."""
    inner = np.conjugate(a1) * a2 + np.conjugate(b1) * b2
    return float(abs(inner) ** 2)
