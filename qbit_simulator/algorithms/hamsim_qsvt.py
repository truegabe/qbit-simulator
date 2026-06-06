"""Hamiltonian simulation via Quantum Signal Processing / QSVT.

For a Hermitian H with eigenvalues in [-1, 1], the time-evolution
operator exp(-iHt) can be expressed as a polynomial in H via the
Jacobi-Anger expansion of cos and sin:

    cos(t · H) = J_0(t) · I + 2 Σ_{k=1}^∞ (-1)^k J_{2k}(t) T_{2k}(H)
    sin(t · H) = 2 Σ_{k=0}^∞ (-1)^k J_{2k+1}(t) T_{2k+1}(H)

with T_k the Chebyshev polynomials of the first kind. Truncating to
degree d ≈ t + log(1/ε) gives error ε. This converts H simulation
into a degree-d polynomial of H — exactly what QSVT (qsvt.py) computes
on a block-encoded H.

This module provides:

  - `truncate_hamsim_polynomial(t, eps)`: pick degree d such that the
    Chebyshev tail is below ε.
  - `hamsim_via_chebyshev(H, t, max_degree)`: build exp(-iHt) directly
    via Chebyshev expansion (without QSVT block-encoding plumbing),
    useful for verification.
  - `hamsim_error_bound(t, d)`: a priori bound on |exp(-iHt) - approx|.
  - `simulate_evolution(H, psi0, t)`: full state evolution.

Compared with Trotter, QSP/QSVT Hamiltonian simulation has OPTIMAL
scaling: query complexity O(t · ||H|| + log(1/ε)) vs Trotter's
O((t · ||H||)² / ε^{1/k}). For long simulations it's a huge win.
"""

from __future__ import annotations

import numpy as np
from scipy.special import jv as _bessel_j


# ----------------------------------------------------------------------------
# Polynomial degree selection
# ----------------------------------------------------------------------------

def truncate_hamsim_polynomial(t: float, eps: float = 1e-8) -> int:
    """Determine the truncation degree d such that the Jacobi-Anger
    tail bound:

        | Σ_{k>d} J_k(t) |  ≤  ε

    Asymptotically, J_k(t) ≈ (t/2)^k / k! for k >> t. We pick d such
    that the next-term magnitude is below ε.
    """
    if t <= 0:
        return 0
    d = max(int(np.ceil(2 * t + np.log(1 / max(eps, 1e-15)))), 1)
    return d


# ----------------------------------------------------------------------------
# Chebyshev-based Hamiltonian simulation
# ----------------------------------------------------------------------------

def hamsim_via_chebyshev(H: np.ndarray, t: float,
                           max_degree: int | None = None,
                           eps: float = 1e-10) -> np.ndarray:
    """Build exp(-iHt) by direct Chebyshev expansion.

    For H with ||H|| ≤ 1, we have

        exp(-iHt) = cos(tH) - i sin(tH)

    and each of cos/sin admits a Chebyshev (Jacobi-Anger) expansion.

    Args:
        H:           Hermitian matrix with ||H||_op ≤ 1.
        t:           evolution time.
        max_degree:  truncation degree (auto if None).
        eps:         target error (used if max_degree is None).

    Returns:
        2^n × 2^n approximation to exp(-iHt).
    """
    if max_degree is None:
        max_degree = truncate_hamsim_polynomial(t, eps)
    d = H.shape[0]
    T_km1 = np.eye(d, dtype=np.complex128)   # T_0(H) = I
    T_k = H.astype(np.complex128).copy()      # T_1(H) = H
    # Coefficients: c_k = J_k(t) for cos/sin parts.
    # exp(-i t H) = J_0(t) I + 2 sum_{k=1}^∞ (-i)^k J_k(t) T_k(H)
    U = _bessel_j(0, t) * T_km1.copy()
    for k in range(1, max_degree + 1):
        if k == 1:
            T_curr = T_k
        else:
            T_curr = 2.0 * H @ T_k - T_km1
            T_km1 = T_k
            T_k = T_curr
        coef = 2.0 * (-1j) ** k * _bessel_j(k, t)
        U = U + coef * T_curr
    return U


# ----------------------------------------------------------------------------
# Error estimate
# ----------------------------------------------------------------------------

def hamsim_error_bound(t: float, d: int) -> float:
    """A-priori bound on Chebyshev-truncation error in Hamiltonian
    simulation: ~ sum_{k>d} 2 |J_k(t)|.

    For large k, |J_k(t)| ≈ (t/2)^k / k! → decays super-exponentially.
    """
    if d < 0:
        return 1.0
    # Tail bound: 2 |J_{d+1}(t)| + (k+1)-th term.
    bound = 2 * abs(_bessel_j(d + 1, t))
    # Geometric-ish tail.
    return float(bound + 2 * abs(_bessel_j(d + 2, t)))


# ----------------------------------------------------------------------------
# Full simulation
# ----------------------------------------------------------------------------

def simulate_evolution(H: np.ndarray, psi0: np.ndarray, t: float,
                         eps: float = 1e-8) -> np.ndarray:
    """Compute exp(-iHt) |ψ_0⟩ via QSP-style Chebyshev expansion.

    Returns the evolved state vector. For H with ||H||_op ≤ 1 the
    approximation is rigorous; otherwise we rescale H → H / ||H|| and
    rescale t accordingly.
    """
    H = np.asarray(H, dtype=np.complex128)
    psi0 = np.asarray(psi0, dtype=np.complex128)
    norm_H = np.linalg.norm(H, ord=2)
    if norm_H == 0:
        return psi0.copy()
    H_normed = H / norm_H
    t_normed = t * norm_H
    U = hamsim_via_chebyshev(H_normed, t_normed, eps=eps)
    return U @ psi0
