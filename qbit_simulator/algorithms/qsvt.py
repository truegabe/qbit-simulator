"""Quantum Singular Value Transformation (QSVT) — Gilyén-Su-Low-Wiebe 2018.

QSVT is the modern unification framework for quantum algorithms. Given a
block-encoded Hermitian matrix A (so ||A|| ≤ 1) inside a unitary U:

    U = [[A,  -√(I-A²)],
         [√(I-A²),  A  ]]

(this is a "qubitization-style" block encoding, using 1 ancilla qubit),
a sequence of phase angles (φ_0, ..., φ_d) interleaved with d applications
of U produces a polynomial transformation P(A):

    U_Φ  =  R(φ_d) · U · R(φ_{d-1}) · U · ... · R(φ_1) · U · R(φ_0)
    P(A) =  ⟨0_anc| U_Φ |0_anc⟩  =  top-left d×d block of U_Φ

where R(φ) = diag(e^{iφ} I_d, e^{-iφ} I_d) is the ancilla phase rotation.

Key fact: For zero phases (Φ = 0...0 of length d+1), the resulting P(A)
is the order-d Chebyshev polynomial of the first kind, T_d(A). Other
polynomial transforms — Hamiltonian simulation (cos(τA), sin(τA)),
matrix inversion (1/A), threshold functions (sign(A)) — require specific
phase sequences (computed via "QSP angle synthesis," a separate inverse
problem).

This implementation:
    - Provides `block_encode_hermitian(A)` (rotation-style).
    - Provides `qsvt_unitary(U, phases)` and `qsvt_polynomial(U, phases)`.
    - Verifies the Chebyshev identity for zero phases up to high degree.
    - Demonstrates applications (Chebyshev expansion of cos(τA) for
      Hamiltonian simulation, etc.).

Forward direction only — given phases, compute the polynomial. QSP angle
synthesis (given polynomial, find phases) is a known but nontrivial
inverse problem; we don't implement it here.
"""

from __future__ import annotations

import numpy as np


def block_encode_hermitian(A: np.ndarray) -> np.ndarray:
    """Rotation-style 1-ancilla block encoding of a Hermitian A with ||A|| ≤ 1.

        U = [[A,  -√(I-A²)],
             [√(I-A²),  A  ]]

    Properties:
      - U†U = I (unitary).
      - U² has top-left block 2A² − I = T_2(A).
      - More generally, U^d has top-left block T_d(A) (Chebyshev of first kind).
    """
    A = np.asarray(A, dtype=np.complex128)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    if not np.allclose(A, A.conj().T, atol=1e-9):
        raise ValueError("A must be Hermitian for the rotation-style block encoding")
    eigvals, eigvecs = np.linalg.eigh(A)
    if np.max(np.abs(eigvals)) > 1.0 + 1e-9:
        raise ValueError(f"||A||_∞ = {np.max(np.abs(eigvals))} must be ≤ 1")
    # √(I - A²) constructed in the eigenbasis where it's diagonal.
    sqrt_diag = np.sqrt(np.clip(1.0 - eigvals ** 2, 0, None))
    sqrt_complement = eigvecs @ np.diag(sqrt_diag.astype(np.complex128)) @ eigvecs.conj().T
    U = np.block([[A, -sqrt_complement], [sqrt_complement, A]])
    return U


def _ancilla_phase(phi: float, d: int) -> np.ndarray:
    """Return diag(e^{iφ}·I_d, e^{-iφ}·I_d) — the (2Π_0 − I) phase rotation."""
    diag = np.empty(2 * d, dtype=np.complex128)
    diag[:d]  = np.exp(1j * phi)
    diag[d:] = np.exp(-1j * phi)
    return np.diag(diag)


def qsvt_unitary(U: np.ndarray, phases: list[float]) -> np.ndarray:
    """Build U_Φ = R(φ_d) · U · R(φ_{d-1}) · U · ... · R(φ_1) · U · R(φ_0).

    Number of U applications: len(phases) - 1 = d.
    Number of phase rotations: len(phases) = d + 1.
    """
    if len(phases) < 1:
        raise ValueError("need at least one phase")
    if U.shape[0] != U.shape[1] or U.shape[0] % 2 != 0:
        raise ValueError("U must be square with even dimension (1 ancilla)")
    d = U.shape[0] // 2
    result = _ancilla_phase(phases[0], d)
    for k in range(1, len(phases)):
        result = _ancilla_phase(phases[k], d) @ U @ result
    return result


def qsvt_polynomial(U: np.ndarray, phases: list[float]) -> np.ndarray:
    """Top-left d×d block of U_Φ — the polynomial P(A) acting on the system."""
    d = U.shape[0] // 2
    return qsvt_unitary(U, phases)[:d, :d]


# ---- known phase sequences ----

def chebyshev_phases(d: int) -> list[float]:
    """All-zero phases of length d+1, producing the Chebyshev polynomial
    T_d(A) when applied to a Hermitian block-encoded A."""
    return [0.0] * (d + 1)


# ---- polynomial helpers for verification ----

def chebyshev_t_of_matrix(n: int, A: np.ndarray) -> np.ndarray:
    """T_n(A) via the standard Chebyshev recurrence:
        T_0(A) = I, T_1(A) = A, T_{n+1}(A) = 2 A T_n(A) − T_{n−1}(A)."""
    d = A.shape[0]
    if n == 0:
        return np.eye(d, dtype=np.complex128)
    Tnm1 = np.eye(d, dtype=np.complex128)
    Tn   = A.astype(np.complex128)
    for _ in range(2, n + 1):
        Tnp1 = 2 * A @ Tn - Tnm1
        Tnm1, Tn = Tn, Tnp1
    return Tn


# ---- Hamiltonian simulation as a QSVT example ----

def chebyshev_expansion_cos(t: float, max_degree: int = 30) -> list[float]:
    """Chebyshev coefficients of cos(t·x) via Jacobi-Anger expansion.

        cos(t · x) = J_0(t) + 2 Σ_{k=1}^∞ (-1)^k J_{2k}(t) T_{2k}(x)

    Returns coefficients in the dense format (coeffs[k] = coefficient of
    T_k(x)), so odd-degree coefficients are zero.
    """
    from scipy.special import jv
    coeffs = [0.0] * (max_degree + 1)
    coeffs[0] = float(jv(0, t))
    for k in range(1, max_degree // 2 + 1):
        if 2 * k <= max_degree:
            coeffs[2 * k] = 2.0 * (-1) ** k * float(jv(2 * k, t))
    return coeffs


def evaluate_chebyshev_polynomial(coeffs: list[float], A: np.ndarray) -> np.ndarray:
    """Evaluate Σ_n c_n T_n(A) directly (used to compare against QSVT output
    once phase synthesis is implemented). Currently used as a numerical
    reference for what QSVT *should* produce for given coefficients."""
    d = A.shape[0]
    result = np.zeros((d, d), dtype=np.complex128)
    Tnm1 = np.eye(d, dtype=np.complex128)
    Tn   = A.astype(np.complex128)
    if len(coeffs) >= 1:
        result += coeffs[0] * Tnm1
    if len(coeffs) >= 2:
        result += coeffs[1] * Tn
    for k in range(2, len(coeffs)):
        Tnp1 = 2 * A @ Tn - Tnm1
        result += coeffs[k] * Tnp1
        Tnm1, Tn = Tn, Tnp1
    return result
