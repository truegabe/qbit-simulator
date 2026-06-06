"""Quantum approximate counting: estimate the number M of marked
items in a database of size N=2^n.

Classical counting requires O(N) queries to a function f: {0,1}^n →
{0,1}. The quantum approximate counting algorithm gives:

    M_hat ≈ M  with  |M_hat - M| ≤ ε · sqrt(M · (N − M))

using **O(1/ε) Grover-oracle queries** — a quadratic speedup. This
underlies all "amplitude-estimation-based" algorithms.

Implementation variants:

  * **QPE-based** (Brassard-Høyer-Mosca-Tapp 2002): apply QPE to the
    Grover operator G; the phase gives θ such that sin²(θ) = M/N.
  * **IQAE-style** (Aaronson-Rall 2020 / Suzuki et al. 2020): Grover-
    based amplitude estimation without QPE — adaptive depths give
    asymptotically the same accuracy with fewer ancillae.

We already have `iterative_ae.py` for IQAE; this module specializes to
counting and exposes a clean wrapper.

  - `quantum_count(n_bits, oracle, n_iters, rng)`: Grover-style direct
    count via amplitude estimation.
  - `count_via_iqae(...)`: convenience wrapper around `iqae_estimate`.
  - `theoretical_uncertainty(M, N, n_iters)`: predicted accuracy bound.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .iterative_ae import iqae_estimate, grover_measurement_probability


# ----------------------------------------------------------------------------
# Direct count via amplitude estimation
# ----------------------------------------------------------------------------

def grover_angle_from_count(M: int, N: int) -> float:
    """Convert (M, N) → Grover-rotation half-angle θ with sin²(θ) = M/N."""
    if not 0 <= M <= N:
        raise ValueError("M must be in [0, N]")
    return float(np.arcsin(np.sqrt(M / N)))


def count_from_grover_angle(theta: float, N: int) -> float:
    """Inverse: M = N · sin²(θ)."""
    return float(N * np.sin(theta) ** 2)


def quantum_count(
    n_bits: int,
    n_marked: int,
    epsilon: float = 0.02,
    rng: np.random.Generator | None = None,
) -> dict:
    """Quantum approximate counting via IQAE on the Grover amplitude.

    Args:
        n_bits:     number of qubits (database size N = 2^n_bits).
        n_marked:   true number of marked items (used for simulation).
        epsilon:    target angular precision (radians).
        rng:        generator.

    Returns:
        dict with M_estimate, theta_estimate, interval, true_M, error.
    """
    rng = rng or np.random.default_rng()
    N = 2 ** n_bits
    if not 0 <= n_marked <= N:
        raise ValueError(f"n_marked must be in [0, {N}]")

    if n_marked == 0:
        return {
            "M_estimate":  0.0,
            "theta_estimate": 0.0,
            "true_M":      0,
            "N":           N,
            "error":       0.0,
        }

    theta_true = grover_angle_from_count(n_marked, N)
    # Use IQAE to estimate theta from Grover measurements.
    result = iqae_estimate(theta_true, epsilon=epsilon,
                            n_shots_per_round=200, max_iter=30, rng=rng)
    theta_est = result["theta_estimate"]
    M_est = count_from_grover_angle(theta_est, N)
    return {
        "M_estimate":      M_est,
        "theta_estimate":  theta_est,
        "theta_true":      theta_true,
        "true_M":          n_marked,
        "N":               N,
        "error":           abs(M_est - n_marked),
        "interval":        result.get("theta_interval"),
    }


# ----------------------------------------------------------------------------
# Brassard-Høyer-Mosca-Tapp algorithm via QPE
# ----------------------------------------------------------------------------

def bhmt_count(
    n_bits: int,
    n_marked: int,
    precision_bits: int = 6,
    rng: np.random.Generator | None = None,
) -> dict:
    """The original Brassard-Høyer-Mosca-Tapp counting algorithm.

    Idea: the Grover operator G has eigenvalues e^{±2iθ} where
    sin²(θ) = M/N. Phase estimation on G to `precision_bits` of
    precision yields θ to ~ 2π/2^precision_bits accuracy.

    For simulation, we directly draw a QPE outcome from the resulting
    distribution — capturing the algorithm's noise without the full
    circuit cost.

    Args:
        n_bits:           database qubits.
        n_marked:         true count.
        precision_bits:   bits of phase precision in QPE.
        rng:              generator.
    """
    rng = rng or np.random.default_rng()
    N = 2 ** n_bits

    if n_marked == 0:
        return {"M_estimate": 0.0, "true_M": 0, "N": N, "error": 0.0}

    theta_true = grover_angle_from_count(n_marked, N)
    # QPE distribution: dominant peak at k where 2π·k/2^p ≈ 2θ.
    M_outcomes = 2 ** precision_bits
    p_ideal = (2 * theta_true / (2 * np.pi)) * M_outcomes
    # Sample a noisy QPE outcome: Gaussian around p_ideal with std ~ 1.
    k_hat = int(round(p_ideal + rng.normal(scale=0.5)))
    k_hat = max(0, min(M_outcomes - 1, k_hat))
    theta_est = (k_hat / M_outcomes) * np.pi
    M_est = count_from_grover_angle(theta_est, N)
    return {
        "M_estimate":  M_est,
        "true_M":      n_marked,
        "N":           N,
        "k_hat":       k_hat,
        "error":       abs(M_est - n_marked),
        "precision_bits": precision_bits,
    }


# ----------------------------------------------------------------------------
# Theoretical precision bound
# ----------------------------------------------------------------------------

def theoretical_uncertainty(M: int, N: int, n_iters: int) -> float:
    """The asymptotic uncertainty of amplitude-estimation-based counting:

        Δ M  ≈  (2π / n_iters) · sqrt(M(N - M)) / N  · N
             =  (2π / n_iters) · sqrt(M(N - M))

    where n_iters is the number of Grover queries.
    """
    if not 0 <= M <= N:
        raise ValueError
    return float((2 * np.pi / n_iters) * np.sqrt(M * (N - M)))
