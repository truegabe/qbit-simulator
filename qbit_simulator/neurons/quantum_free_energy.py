"""Quantum free-energy bridge — extended.

The classical Free Energy F = E - T S has a direct quantum analogue:

    F = ⟨H⟩ - T · S(ρ)

with S(ρ) = -Tr(ρ log ρ) the von Neumann entropy. The Gibbs state
ρ_β = exp(-β H) / Z minimizes F at temperature 1/β.

In Friston's active inference, the brain minimizes variational free
energy F[q] = ⟨log q − log p⟩_q. Replacing the classical KL by quantum
relative entropy gives the same structure: a quantum belief state ρ
should approximate the "true" Gibbs posterior σ.

This module gives:
  - Quantum relative entropy S(ρ || σ) = Tr ρ (log ρ - log σ).
  - Variational Bayes on a quantum belief: minimize S(ρ || σ_β).
  - Bridge to classical predictive coding (returns the matching
    classical F up to a constant).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def safe_log_matrix(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Matrix log via eigendecomposition (Hermitian A)."""
    evals, evecs = np.linalg.eigh(A)
    evals = np.maximum(evals, eps)
    return evecs @ np.diag(np.log(evals)) @ evecs.conj().T


def von_neumann_entropy(rho: np.ndarray) -> float:
    evals = np.linalg.eigvalsh(rho)
    evals = np.maximum(evals, 0)
    s = 0.0
    for e in evals:
        if e > 1e-12:
            s -= e * np.log(e)
    return float(s)


def quantum_relative_entropy(rho: np.ndarray, sigma: np.ndarray) -> float:
    """S(ρ || σ) = Tr ρ (log ρ - log σ).  Ill-defined if supp(ρ) ⊄ supp(σ)."""
    return float(np.real(np.trace(rho @ (safe_log_matrix(rho)
                                          - safe_log_matrix(sigma)))))


def gibbs_state(H: np.ndarray, beta: float = 1.0) -> np.ndarray:
    evals, evecs = np.linalg.eigh(H)
    e = np.exp(-beta * (evals - evals.min()))
    Z = e.sum()
    return evecs @ np.diag(e / Z) @ evecs.conj().T


def quantum_free_energy(rho: np.ndarray, H: np.ndarray,
                          beta: float = 1.0) -> float:
    """F[ρ] = ⟨H⟩ - (1/β) S(ρ)."""
    energy = float(np.real(np.trace(rho @ H)))
    return energy - (1.0 / beta) * von_neumann_entropy(rho)


def classical_free_energy(p: np.ndarray, E: np.ndarray,
                           beta: float = 1.0) -> float:
    """Classical F = sum p E - (1/β) H(p)."""
    p = np.clip(p, 1e-12, None)
    H = -(p * np.log(p)).sum()
    return float((p * E).sum() - (1.0 / beta) * H)
