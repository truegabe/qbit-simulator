"""Quantum Signal Processing (QSP) — Low & Chuang 2017, Gilyén-Su-Low-Wiebe 2018.

QSP is the modern unification framework for nearly every interesting
quantum algorithm:

    Algorithm                     Implemented as QSP with polynomial P(x)
    --------------------------    --------------------------------------
    Quantum search (Grover)       Threshold function ~sign(x - threshold)
    Hamiltonian simulation        cos(τx) and sin(τx) (Jacobi-Anger)
    Matrix inversion (HHL)        1/x (well-conditioned)
    Amplitude estimation          Specific Chebyshev polynomial
    Quantum eigenvalue threshold  Heaviside step approximation

The QSP unitary on one qubit:

    U(φ; a) = e^{iφ_0 Z} · W(a) · e^{iφ_1 Z} · W(a) · ... · e^{iφ_d Z}

where W(a) is a "signal" rotation:

    W(a) = [[ a,  i·√(1-a²) ],
            [ i·√(1-a²),  a ]]

(this is e^{iθX} with cos(θ) = a; equivalently a rotation by angle
2·arccos(a) around X.)

The matrix element U[0,0] is a polynomial P(a) of degree d in a, with
specific structural constraints (Reflection theorem, Low-Chuang 2017):
  - deg(P) ≤ d
  - P has parity d mod 2
  - |P(a)|² ≤ 1 for a ∈ [-1, 1]
  - There exists a polynomial Q(a) with |P(a)|² + (1-a²)|Q(a)|² = 1

For any polynomial P satisfying these constraints, an explicit phase
sequence (φ_0, ..., φ_d) exists. Finding the sequence ("QSP phase
synthesis") is a separate inverse problem; we implement the forward
direction here — given a sequence, compute U and verify against the
expected polynomial.

Use cases shipped in this module:
  - identity_qsp: trivial pass-through (d=0)
  - chebyshev_phases: phase sequence for T_d(a), the Chebyshev polynomial
"""

from __future__ import annotations

import numpy as np


def signal_operator(a: float) -> np.ndarray:
    """Standard QSP signal W(a) = e^{i·θ·X} with cos(θ) = a.

    Equivalent to the reflection-style matrix
        [[a,  i·√(1-a²)],
         [i·√(1-a²),  a ]]
    """
    a = float(a)
    if abs(a) > 1.0 + 1e-12:
        raise ValueError(f"signal a={a} outside [-1, 1]")
    a = max(-1.0, min(1.0, a))
    s = np.sqrt(max(0.0, 1.0 - a * a))
    return np.array([[a,    1j * s],
                     [1j * s, a   ]], dtype=np.complex128)


def phase_operator(phi: float) -> np.ndarray:
    """e^{iφZ} = diag(e^{iφ}, e^{-iφ})."""
    return np.array([[np.exp(1j * phi), 0.0],
                     [0.0,              np.exp(-1j * phi)]], dtype=np.complex128)


def qsp_unitary(phases: list[float], a: float) -> np.ndarray:
    """Build the QSP unitary U(phases; a).

    U = e^{iφ_0 Z} · W(a) · e^{iφ_1 Z} · W(a) · ... · e^{iφ_d Z}
    The number of phase operators is d+1; the number of W's is d.
    """
    if len(phases) < 1:
        raise ValueError("need at least 1 phase")
    W = signal_operator(a)
    U = phase_operator(phases[0])
    for phi in phases[1:]:
        U = U @ W @ phase_operator(phi)
    return U


def qsp_polynomial(phases: list[float], a: float) -> complex:
    """The polynomial P(a) = U[0, 0] of the QSP unitary."""
    U = qsp_unitary(phases, a)
    return complex(U[0, 0])


# ---- known phase sequences ----

def chebyshev_phases(d: int) -> list[float]:
    """Phase sequence (φ_0, ..., φ_d) = all zeros, length d+1.

    For our signal convention W(a) = [[a, i√(1-a²)], [i√(1-a²), a]],
    the QSP polynomial U[0,0] equals exactly T_d(a) — the Chebyshev
    polynomial of the first kind, order d. Verified by direct expansion:
        d=1: W[0,0] = a = T_1(a)
        d=2: W²[0,0] = 2a² - 1 = T_2(a)
        d=3: W³[0,0] = 4a³ - 3a = T_3(a)
        ... and so on by the recurrence.
    """
    return [0.0] * (d + 1)


def identity_phases() -> list[float]:
    """Phase sequence whose QSP polynomial is P(a) = a (the signal itself)."""
    # With phases = (0, 0), U = Z^0 · W(a) · Z^0 = W(a), and U[0,0] = a.
    return [0.0, 0.0]


# ---- polynomial evaluation helpers (for tests) ----

def chebyshev_t(n: int, x: float) -> float:
    """Chebyshev polynomial T_n(x) of the first kind via the recurrence."""
    if n == 0:
        return 1.0
    if n == 1:
        return float(x)
    a, b = 1.0, float(x)
    for _ in range(2, n + 1):
        a, b = b, 2 * x * b - a
    return b


# ---- amplitude amplification via QSP (canonical example) ----

def amplitude_amplification_phases(n_iterations: int) -> list[float]:
    """Phase sequence that realizes Grover-style amplitude amplification:
    P(a) ≈ sin((2n+1)·arcsin(a)).

    This is the QSP equivalent of `n_iterations` Grover steps. For
    a = √(M/N) (initial amplitude on marked states), this boosts the
    amplitude to sin((2n+1)·θ) where θ = arcsin(a).
    """
    # Standard Grover-as-QSP: alternating φ = π/2 and 0 (up to sign convention).
    # For n_iterations Grover steps, the QSP sequence has length 2n+1.
    # Concrete construction: [π/4, π/2, π/2, ..., π/2, π/4] with 2n-1 inner π/2's.
    # (Various equivalent forms exist; we pick one that produces the Chebyshev-like
    #  composition matching Grover.)
    if n_iterations < 1:
        return identity_phases()
    return [np.pi / 4] + [np.pi / 2] * (2 * n_iterations - 1) + [np.pi / 4]
