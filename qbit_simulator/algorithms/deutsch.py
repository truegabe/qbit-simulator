"""Deutsch algorithm: decide if f:{0,1}->{0,1} is constant or balanced in 1 query.

We represent f via its oracle U_f acting on |x>|y> -> |x>|y XOR f(x)>.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..circuit import QuantumCircuit


def _oracle_matrix(f: Callable[[int], int]) -> np.ndarray:
    U = np.zeros((4, 4), dtype=np.complex128)
    for x in (0, 1):
        for y in (0, 1):
            inp = (x << 1) | y
            out = (x << 1) | (y ^ f(x))
            U[out, inp] = 1.0
    return U


def deutsch(f: Callable[[int], int]) -> str:
    """Returns 'constant' or 'balanced'."""
    qc = QuantumCircuit(2)
    qc.x(1)          # |0,1>
    qc.h(0).h(1)     # |+,->
    qc._apply_2q(_oracle_matrix(f), 0, 1)
    qc.h(0)
    p = qc.probabilities()
    # Marginal probability of q0 == 0.
    # Index bits: state index = q0*2 + q1, so q0==0 -> indices 0,1.
    p_q0_zero = p[0] + p[1]
    return "constant" if p_q0_zero > 0.5 else "balanced"
