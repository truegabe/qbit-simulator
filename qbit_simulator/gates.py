"""Quantum gates as complex matrices."""

from __future__ import annotations

import numpy as np

_SQRT2_INV = 1 / np.sqrt(2)

I2 = np.array([[1, 0], [0, 1]], dtype=np.complex128)

H = _SQRT2_INV * np.array([[1, 1], [1, -1]], dtype=np.complex128)

X = np.array([[0, 1], [1, 0]], dtype=np.complex128)

Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)

Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)

T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)

# Full 4x4 CNOT with qubit 0 as control, qubit 1 as target (basis order |q0 q1>).
CNOT = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ],
    dtype=np.complex128,
)

SWAP = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.complex128,
)


# --- parameterized single-qubit gates ---

def Rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def Ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def Rz(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
        dtype=np.complex128,
    )


def P(phi: float) -> np.ndarray:
    """Phase gate: |0> -> |0>, |1> -> e^{i phi}|1>."""
    return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=np.complex128)


# --- two-qubit constructors ---

def controlled(U: np.ndarray) -> np.ndarray:
    """Build a 4x4 controlled-U with control as MSB (q0) and target as LSB (q1)."""
    out = np.eye(4, dtype=np.complex128)
    out[2:, 2:] = U
    return out


def CP(phi: float) -> np.ndarray:
    return controlled(P(phi))


def is_unitary(m: np.ndarray, tol: float = 1e-10) -> bool:
    n = m.shape[0]
    return np.allclose(m.conj().T @ m, np.eye(n), atol=tol)
