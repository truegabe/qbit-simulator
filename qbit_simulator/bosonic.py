"""Bosonic / continuous-variable error correction: cat and GKP codes.

In quantum hardware, the underlying physical degree of freedom is often
a bosonic mode (a resonator, transmon, or trapped-ion motional mode)
with an infinite-dimensional Hilbert space. Encoding a qubit in a single
bosonic mode — rather than in many physical qubits — is sometimes more
hardware-efficient. Two canonical bosonic codes:

  * **Cat codes** (Mirrahimi et al. 2014):
        |0_L⟩ = N (|α⟩ + |−α⟩)   (even-parity cat)
        |1_L⟩ = N (|α⟩ − |−α⟩)   (odd-parity cat)
    The two logicals differ by photon-number parity; photon loss
    (a |α⟩ → α |α⟩) flips between them, so parity measurement is the
    natural syndrome.

  * **GKP codes** (Gottesman-Kitaev-Preskill 2001):
        |0_L⟩ = sum_n |x = (2n) · √π⟩
        |1_L⟩ = sum_n |x = (2n+1) · √π⟩
    A lattice in phase space; protects against small shifts in both x
    and p simultaneously. The natural syndrome is measuring x mod √π.

This module simulates these codes on a TRUNCATED Fock space of
dimension D (typically 20–30 for cat states with α ≤ 3, larger for
GKP). Provides:

  - `coherent_state(alpha, D)`: |α⟩ in the Fock basis.
  - `cat_state(alpha, parity, D)`: |0_L⟩ or |1_L⟩.
  - `photon_loss(rho, gamma, D)`: apply the photon-loss channel for
    time interval t with rate gamma (γ = 1 − e^{−κt}).
  - `cat_parity_syndrome(rho)`: ⟨(−1)^n⟩ — the parity expectation.
  - `gkp_state(parity, sigma, n_terms, D)`: approximate GKP |0_L⟩ or
    |1_L⟩ as a normalized Gaussian envelope on the position grid.
  - `displacement(alpha, D)`: D(α) = exp(α a† − α* a).
  - `position_eigenvalue_expectation(rho)`: ⟨x⟩.

The math is exact (no Monte Carlo): we work with density matrices on
the truncated Fock space and apply analytic channels.
"""

from __future__ import annotations

from math import factorial

import numpy as np
from scipy.linalg import expm


# ----------------------------------------------------------------------------
# Fock-space operators
# ----------------------------------------------------------------------------

def annihilation_operator(D: int) -> np.ndarray:
    """The bosonic annihilation a in the Fock basis truncated to dimension D.

        a |n⟩ = √n |n-1⟩
    """
    a = np.zeros((D, D), dtype=np.complex128)
    for n in range(1, D):
        a[n - 1, n] = np.sqrt(n)
    return a


def creation_operator(D: int) -> np.ndarray:
    """a† in the truncated Fock basis."""
    return annihilation_operator(D).conj().T


def number_operator(D: int) -> np.ndarray:
    """N = a† a (diagonal with entries 0, 1, 2, ..., D-1)."""
    return np.diag(np.arange(D)).astype(np.complex128)


def position_operator(D: int) -> np.ndarray:
    """x = (a + a†) / √2."""
    a = annihilation_operator(D)
    return (a + a.conj().T) / np.sqrt(2)


def momentum_operator(D: int) -> np.ndarray:
    """p = -i(a − a†) / √2."""
    a = annihilation_operator(D)
    return -1j * (a - a.conj().T) / np.sqrt(2)


# ----------------------------------------------------------------------------
# Coherent and cat states
# ----------------------------------------------------------------------------

def coherent_state(alpha: complex, D: int) -> np.ndarray:
    """|α⟩ = e^{−|α|²/2} Σ_n (α^n / √n!) |n⟩ in the Fock basis."""
    psi = np.zeros(D, dtype=np.complex128)
    log_factorial = 0.0
    for n in range(D):
        if n > 0:
            log_factorial += np.log(n)
        # log of α^n / sqrt(n!) = n log(α) - 0.5 * log(n!)
        # Better to compute incrementally to avoid overflow.
        if n == 0:
            psi[n] = 1.0
        else:
            psi[n] = psi[n - 1] * alpha / np.sqrt(n)
    psi *= np.exp(-0.5 * abs(alpha) ** 2)
    return psi


def displacement_operator(alpha: complex, D: int) -> np.ndarray:
    """D(α) = exp(α a† − α* a)."""
    a = annihilation_operator(D)
    return expm(alpha * a.conj().T - np.conj(alpha) * a)


def cat_state(alpha: float, parity: int = 0, D: int = 30) -> np.ndarray:
    """Cat state with mean photon number ≈ |α|² and given photon-number parity.

    Args:
        alpha:   real positive amplitude.
        parity:  0 → even (|0_L⟩), 1 → odd (|1_L⟩).
        D:       Fock-space cutoff. For α ≤ 3, D = 30 is safe.
    """
    psi_plus = coherent_state(alpha, D)
    psi_minus = coherent_state(-alpha, D)
    if parity == 0:
        psi = psi_plus + psi_minus
    elif parity == 1:
        psi = psi_plus - psi_minus
    else:
        raise ValueError("parity must be 0 or 1")
    psi /= np.linalg.norm(psi)
    return psi


def cat_parity(psi_or_rho: np.ndarray) -> float:
    """⟨(−1)^n⟩ — the photon-number-parity expectation value.

    Equal to +1 for |0_L⟩ (even cat), −1 for |1_L⟩ (odd cat).
    """
    D = psi_or_rho.shape[0]
    parity_op = np.diag([(-1) ** n for n in range(D)]).astype(np.complex128)
    if psi_or_rho.ndim == 1:
        return float(np.real(psi_or_rho.conj() @ parity_op @ psi_or_rho))
    return float(np.real(np.trace(parity_op @ psi_or_rho)))


# ----------------------------------------------------------------------------
# Photon-loss channel
# ----------------------------------------------------------------------------

def photon_loss(rho: np.ndarray, gamma: float) -> np.ndarray:
    """Apply the photon-loss channel: rho → Σ_k E_k rho E_k†

    with Kraus operators E_k = √(γ^k / k!) · a^k · (1−γ)^{N/2}.

    γ ∈ [0, 1] is the per-time-step loss probability (γ = 1 − exp(−κ t)).
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    D = rho.shape[0]
    a = annihilation_operator(D)
    N = number_operator(D)
    # (1 - γ)^{N/2}
    decay = np.diag([(1 - gamma) ** (n / 2.0) for n in range(D)]).astype(np.complex128)
    out = np.zeros_like(rho)
    a_k = np.eye(D, dtype=np.complex128)
    for k in range(D):
        # Correct Kraus operator: E_k = sqrt(γ^k / k!) · (1-γ)^{N/2} · a^k.
        # Decay must be applied AFTER a^k (matrices act right-to-left).
        E_k = np.sqrt(gamma ** k / factorial(k)) * (decay @ a_k)
        out += E_k @ rho @ E_k.conj().T
        a_k = a_k @ a   # increment to a^(k+1)
    return out


# ----------------------------------------------------------------------------
# GKP states (approximate, finite-energy)
# ----------------------------------------------------------------------------

def gkp_state(parity: int = 0, sigma: float = 0.3, n_terms: int = 10,
               D: int = 100) -> np.ndarray:
    """Approximate finite-energy GKP code state.

    |0_L⟩ ∝ Σ_n e^{−(2n √π)² / (4σ²)} D(2n √π / √2) |0⟩
    |1_L⟩ ∝ Σ_n e^{−((2n+1) √π)² / (4σ²)} D((2n+1) √π / √2) |0⟩

    (factor of √2 from x = (a + a†)/√2 conversion of position lattice).

    Args:
        parity:   0 (|0_L⟩) or 1 (|1_L⟩).
        sigma:    width of the Gaussian envelope (smaller = sharper peaks
                  but more energy).
        n_terms:  number of lattice points on each side of 0.
        D:        Fock cutoff. GKP needs large D (~100 for typical params).
    """
    if parity not in (0, 1):
        raise ValueError("parity must be 0 or 1")
    psi = np.zeros(D, dtype=np.complex128)
    vac = np.zeros(D, dtype=np.complex128)
    vac[0] = 1.0
    sqrt_pi = np.sqrt(np.pi)
    for n in range(-n_terms, n_terms + 1):
        x_n = (2 * n + parity) * sqrt_pi
        # Convert position shift to displacement amplitude:
        # |x = x_n⟩ ≈ D(x_n / √2) |0⟩  (in physicist convention).
        alpha = x_n / np.sqrt(2)
        weight = np.exp(-(x_n ** 2) / (4 * sigma ** 2))
        if weight < 1e-12:
            continue
        D_op = displacement_operator(alpha, D)
        psi += weight * (D_op @ vac)
    psi /= np.linalg.norm(psi)
    return psi


def gkp_x_expectation(psi: np.ndarray) -> float:
    """⟨x⟩ — should be 0 for either GKP logical (lattice is symmetric)."""
    D = psi.shape[0]
    x = position_operator(D)
    return float(np.real(psi.conj() @ x @ psi))


def gkp_x_mod_sqrt_pi(psi: np.ndarray) -> float:
    """⟨cos(2√π · x)⟩ — the natural GKP stabilizer expectation.

    For ideal |0_L⟩ or |1_L⟩, this should be near +1 (the lattice is
    aligned with the cosine). For random states it's ≈ 0.
    """
    D = psi.shape[0]
    x = position_operator(D)
    # cos(2√π · x) via Taylor isn't efficient; use eigendecomp.
    eigs, V = np.linalg.eigh(x)
    op = V @ np.diag(np.cos(2 * np.sqrt(np.pi) * eigs)) @ V.conj().T
    return float(np.real(psi.conj() @ op @ psi))
