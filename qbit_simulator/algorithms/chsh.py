"""CHSH game — Bell inequality violation demonstration.

A referee gives Alice a bit x ∈ {0,1} and Bob a bit y ∈ {0,1}.
Alice outputs a ∈ {0,1}, Bob outputs b ∈ {0,1}.
They win iff a XOR b = x AND y.

Classical optimal strategy: always output 0. Wins 3/4 of the time.

Quantum optimal strategy (sharing a Bell pair):
  Alice's measurement bases:  x=0 → Z,    x=1 → X.
  Bob's measurement bases:    y=0 → (X+Z)/√2, y=1 → (X-Z)/√2.

This wins with probability cos²(π/8) ≈ 0.8536, beating the classical
bound by 10.4 percentage points. This is **Tsirelson's bound** for the
CHSH inequality.

Reference: Clauser, Horne, Shimony, Holt — Phys. Rev. Lett. 23, 880 (1969).
"""

from __future__ import annotations

import numpy as np

from ..circuit import QuantumCircuit
from ..gates import H, X, Z, I2, Ry


def _rotate_basis_for_alice(qc: QuantumCircuit, x: int) -> None:
    """Rotate Alice's qubit (q0) so that a Z-basis measurement implements
    Z for x=0 and X for x=1."""
    if x == 1:
        qc.h(0)


def _rotate_basis_for_bob(qc: QuantumCircuit, y: int) -> None:
    """Rotate Bob's qubit (q1) so that a Z-basis measurement implements
    (X+Z)/√2 for y=0 and (X-Z)/√2 for y=1."""
    # Eigenbasis of (X+Z)/√2 is Z rotated by π/4; for (X-Z)/√2 it's -π/4.
    # Ry(-θ) brings the desired eigenstate onto |0⟩ for Z-basis measurement.
    angle = -np.pi / 4 if y == 0 else np.pi / 4
    qc.ry(angle, 1)


def play_round(x: int, y: int, rng: np.random.Generator) -> tuple[int, int]:
    """Play one round of the CHSH game with the quantum strategy.
    Returns (a, b)."""
    qc = QuantumCircuit(2)
    # Shared Bell pair.
    qc.h(0); qc.cnot(0, 1)
    _rotate_basis_for_alice(qc, x)
    _rotate_basis_for_bob(qc, y)
    a = qc.measure_qubit(0, rng=rng)
    b = qc.measure_qubit(1, rng=rng)
    return a, b


def chsh_quantum_win_rate(n_rounds: int = 4000, rng: np.random.Generator | None = None) -> float:
    rng = rng or np.random.default_rng()
    wins = 0
    for _ in range(n_rounds):
        x = int(rng.integers(0, 2))
        y = int(rng.integers(0, 2))
        a, b = play_round(x, y, rng)
        if (a ^ b) == (x & y):
            wins += 1
    return wins / n_rounds


def chsh_classical_win_rate(n_rounds: int = 4000, rng: np.random.Generator | None = None) -> float:
    """Classical best strategy: both always output 0.
    Wins on (x,y) ∈ {(0,0),(0,1),(1,0)} — 3/4 of cases.
    """
    rng = rng or np.random.default_rng()
    wins = 0
    for _ in range(n_rounds):
        x = int(rng.integers(0, 2))
        y = int(rng.integers(0, 2))
        a, b = 0, 0  # always output 0
        if (a ^ b) == (x & y):
            wins += 1
    return wins / n_rounds


def tsirelson_bound() -> float:
    """Maximum CHSH win probability achievable by any quantum strategy."""
    return float(np.cos(np.pi / 8) ** 2)
