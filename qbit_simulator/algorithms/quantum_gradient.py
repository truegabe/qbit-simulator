"""Jordan's quantum gradient algorithm (Jordan 2005).

Given an oracle that encodes a real-valued function f: [0, 1)^d → R as
a relative phase, Jordan's algorithm estimates the gradient ∇f at a
fixed point in a SINGLE oracle query — vs. d + 1 classical queries.

Protocol
--------
Let f be such that f(x) is well-approximated by its first-order Taylor
expansion around x₀ on the relevant region. Encode each coordinate
x_i into b qubits as an integer y_i ∈ {0, ..., 2^b - 1} so that
x_i = (y_i / 2^b) - 1/2 ∈ [-1/2, 1/2). Prepare each of the d registers
in the uniform superposition (apply Hadamard), giving

    |ψ_0⟩ = (1/2^{b d/2}) Σ_{y ∈ {0,..,2^b−1}^d} |y⟩.

Apply the phase oracle U_f that maps

    |y⟩ → exp(2π i · M · f(x(y))) |y⟩

for an integer M chosen so the phase is sensitive to the gradient.
A first-order Taylor expansion of f gives

    M · f(x(y)) ≈ M · f(x₀) + Σ_i (M / 2^b)(y_i − 2^{b-1}) ∂_i f.

The constant phase factors out; what's left is a tensor-product of
single-register QFT-style states. Applying inverse QFT to each register
reads off (∂_i f) directly.

Result: one query, one inverse QFT per register, recover ∇f.

Implementation (numpy, fully dense)
-----------------------------------
For tractable demos we work in d = 1 or 2 dimensions with b = 4–6 bits
per coordinate. The phase oracle is just `exp(2πiM·f(x))` on each grid
point — built once as a diagonal. Inverse QFT is applied as numpy
matrix multiply.

Returns
-------
- `quantum_gradient(f, x0, d, bits, M, step)`: estimated gradient at x0.
- `phase_oracle_diagonal(...)`: the diagonal we build internally.
"""

from __future__ import annotations

import math

import numpy as np


def _qft_matrix(n: int) -> np.ndarray:
    """n × n Fourier matrix (forward QFT, no factor of 1/√n in exp)."""
    omega = np.exp(2j * np.pi / n)
    return np.array([[omega ** (j * k) for k in range(n)] for j in range(n)]) / np.sqrt(n)


def _iqft_matrix(n: int) -> np.ndarray:
    return _qft_matrix(n).conj().T


def phase_oracle_diagonal(f, x0: np.ndarray, d: int, bits: int,
                            M: float, step: float = 1.0) -> np.ndarray:
    """Build the diagonal of the phase oracle exp(2πiM f(x)) over the grid.

    Grid: each coordinate y_i ∈ {0,..,2^b−1} represents
        x_i  =  x0_i + step * ((y_i / 2^b) − 1/2).
    """
    N = 2 ** bits
    diag = np.zeros(N ** d, dtype=np.complex128)
    idx = 0
    grid = [(y / N - 0.5) for y in range(N)]
    if d == 1:
        for y0 in range(N):
            xi = x0 + step * np.array([grid[y0]])
            phase = 2 * np.pi * M * float(f(xi))
            diag[y0] = np.exp(1j * phase)
        return diag
    if d == 2:
        for y0 in range(N):
            for y1 in range(N):
                xi = x0 + step * np.array([grid[y0], grid[y1]])
                phase = 2 * np.pi * M * float(f(xi))
                diag[y0 * N + y1] = np.exp(1j * phase)
        return diag
    raise NotImplementedError("d > 2 not supported in this demo.")


def quantum_gradient(f, x0: np.ndarray, bits: int = 6,
                      M: float = 16.0, step: float = 0.1) -> dict:
    """Estimate ∇f(x0) using Jordan's algorithm.

    Args:
        f:    callable x (ndarray) → float.
        x0:   point at which to estimate the gradient.
        bits: bits per coordinate (grid resolution = 2^bits).
        M:    phase-scaling constant. Should satisfy M·step·|∂f| ≪ 2^bits
              (so the phase wraps cleanly within one QFT period).
        step: half-width of the local box (must be small enough that
              f is approximately linear inside).

    Returns dict with 'gradient', 'classical_gradient' (finite-diff
    reference), and 'rel_error'.
    """
    x0 = np.asarray(x0, dtype=np.float64)
    d = len(x0)
    if d not in (1, 2):
        raise NotImplementedError("d > 2 not supported in this demo.")
    N = 2 ** bits
    # Initial state: uniform over the d-register N^d grid.
    psi = np.ones(N ** d, dtype=np.complex128) / np.sqrt(N ** d)
    # Apply phase oracle.
    diag = phase_oracle_diagonal(f, x0, d, bits, M, step=step)
    psi = diag * psi
    # Inverse-QFT each register.
    IQFT = _iqft_matrix(N)
    if d == 1:
        psi = IQFT @ psi
    else:
        psi = psi.reshape(N, N)
        psi = IQFT @ psi @ IQFT.T   # IQFT on both indices
        psi = psi.reshape(N ** d)
    probs = np.abs(psi) ** 2
    # Find most-likely outcome per register.
    if d == 1:
        k = int(np.argmax(probs))
        signed = k if k < N // 2 else k - N
        gradient = np.array([signed / (M * step)])
    else:
        idx = int(np.argmax(probs))
        k0 = idx // N
        k1 = idx % N
        s0 = k0 if k0 < N // 2 else k0 - N
        s1 = k1 if k1 < N // 2 else k1 - N
        gradient = np.array([s0, s1]) / (M * step)
    # Classical finite-diff reference.
    h = step * 1e-3
    classical = np.zeros(d)
    for i in range(d):
        e = np.zeros(d); e[i] = h
        classical[i] = (f(x0 + e) - f(x0 - e)) / (2 * h)
    err = float(np.linalg.norm(gradient - classical)
                / (np.linalg.norm(classical) + 1e-12))
    return {"gradient": gradient, "classical_gradient": classical,
            "rel_error": err}
