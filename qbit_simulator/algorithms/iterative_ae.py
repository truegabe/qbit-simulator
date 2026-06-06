"""Iterative Amplitude Estimation (IQAE / MLAE) — Grinko et al. 2021.

Canonical Amplitude Estimation (QAE) requires a counting register of t
ancilla qubits to reach precision 2^-t. For NISQ hardware, this overhead
is too expensive. **Iterative Amplitude Estimation** does the same job with
ONE ancilla, at the cost of more circuit shots — adaptive depth instead
of adaptive width.

Two flavors implemented:

**Maximum Likelihood Amplitude Estimation (MLAE)**:
    Choose a schedule of Grover-operator powers k_0 < k_1 < ... < k_J.
    For each depth k_j, run N_j shots and count "good" outcomes m_j.
    Build the likelihood L(θ) = ∏_j P(m_j | N_j, k_j, θ) and find its
    argmax. Returns the MLE estimate of θ (and hence a = sin²θ).

**Iterative Amplitude Estimation (IQAE)** (Grinko et al.):
    Adaptive bisection on the angle interval θ ∈ [0, π/2]. Each iteration
    picks a Grover depth k that bisects the current interval, runs shots
    to determine which sub-interval θ falls in, narrows the interval,
    and repeats. Converges to precision ε in O(1/ε) total Grover queries
    (compared to O(1/ε²) for classical sampling).

Both methods reproduce the canonical QAE accuracy without the t-qubit
counting register — practical for NISQ-era amplitude estimation.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import minimize_scalar


def grover_measurement_probability(theta: float, k: int) -> float:
    """P(measure "good") for a Grover-iterated state at depth k.

    After applying the Grover operator k times to the state with initial
    amplitude sin(θ), the new amplitude on the marked subspace is
    sin((2k+1)θ). So P(good) = sin²((2k+1)θ).
    """
    return float(np.sin((2 * k + 1) * theta) ** 2)


def sample_grover_outcomes(
    theta_true: float, k: int, n_shots: int,
    rng: np.random.Generator,
) -> int:
    """Simulate n_shots measurements of the Grover-iterated state.

    Returns the number of "good" outcomes.
    """
    p_good = grover_measurement_probability(theta_true, k)
    p_good = float(np.clip(p_good, 0.0, 1.0))
    return int(rng.binomial(n_shots, p_good))


def mlae_estimate(
    theta_true: float,
    grover_depths: list[int],
    n_shots_per_depth: int = 100,
    rng: np.random.Generator | None = None,
) -> dict:
    """Maximum Likelihood Amplitude Estimation.

    Given the true θ (simulated unknown), sample measurements at each
    Grover depth, then find the θ maximizing the likelihood.

    Args:
        theta_true:        unknown phase to estimate. We "know" it for the
                           sampler but the MLE doesn't see it.
        grover_depths:     list of k values to query.
        n_shots_per_depth: shots per depth k.
        rng:               numpy generator.

    Returns:
        dict with theta_estimate, amplitude_estimate (sin² θ_estimate),
        true_theta, true_amplitude.
    """
    rng = rng or np.random.default_rng()
    # Sample at each depth.
    counts = []
    for k in grover_depths:
        m = sample_grover_outcomes(theta_true, k, n_shots_per_depth, rng)
        counts.append((k, m))

    # Build negative log-likelihood as a function of θ.
    def neg_log_lik(theta):
        nll = 0.0
        for k, m in counts:
            p = grover_measurement_probability(theta, k)
            p = max(min(p, 1.0 - 1e-12), 1e-12)
            nll -= m * np.log(p) + (n_shots_per_depth - m) * np.log(1 - p)
        return nll

    # The likelihood has many local minima because Grover at depth k>0
    # makes sin²((2k+1)θ) oscillate. Use a dense grid search to bracket the
    # global minimum, then refine with scalar optimization.
    max_k = max(k for k, _ in counts)
    # Resolution must capture the fastest oscillation: ~ π / (2·max_k).
    n_grid = max(200, 20 * (max_k + 1))
    theta_grid = np.linspace(1e-6, np.pi / 2 - 1e-6, n_grid)
    nll_grid = np.array([neg_log_lik(t) for t in theta_grid])
    best_idx = int(np.argmin(nll_grid))
    # Refine around the grid minimum.
    lo = theta_grid[max(0, best_idx - 1)]
    hi = theta_grid[min(n_grid - 1, best_idx + 1)]
    result = minimize_scalar(neg_log_lik, bounds=(lo, hi),
                              method="bounded", options={"xatol": 1e-7})
    theta_est = float(result.x)
    return {
        "theta_estimate":     theta_est,
        "amplitude_estimate": float(np.sin(theta_est) ** 2),
        "true_theta":         theta_true,
        "true_amplitude":     float(np.sin(theta_true) ** 2),
        "counts":             counts,
    }


def iqae_estimate(
    theta_true: float,
    epsilon: float = 0.01,
    confidence: float = 0.95,
    n_shots_per_round: int = 100,
    max_iter: int = 50,
    rng: np.random.Generator | None = None,
) -> dict:
    """Iterative Amplitude Estimation (simplified Grinko et al. protocol).

    Adaptive: maintain a confidence interval [θ_lo, θ_hi] on the angle.
    Pick the largest Grover depth k whose oscillation fits in the
    current interval, run shots, update via Chernoff bound, repeat.

    Args:
        theta_true:         unknown phase to estimate.
        epsilon:            target precision on θ.
        confidence:         confidence level (e.g. 0.95).
        n_shots_per_round:  shots per iteration.
        max_iter:           safety cap.
        rng:                numpy generator.
    """
    rng = rng or np.random.default_rng()
    theta_lo = 0.0
    theta_hi = np.pi / 2
    n_queries_total = 0
    history = []

    for it in range(max_iter):
        if (theta_hi - theta_lo) / 2 < epsilon:
            break
        # Choose k as the largest integer such that the oscillation
        # period (2k+1)·θ stays monotonic within [θ_lo, θ_hi]. Roughly,
        # 2(2k+1)·(θ_hi - θ_lo) < π → k < (π / (θ_hi - θ_lo) - 2) / 4.
        width = theta_hi - theta_lo
        if width < 1e-9:
            break
        k = max(0, int((np.pi / width - 2) // 4))
        # Sample.
        m = sample_grover_outcomes(theta_true, k, n_shots_per_round, rng)
        n_queries_total += n_shots_per_round * (2 * k + 1)
        history.append((k, m))
        # Update interval via Chernoff-style bound on the binomial estimate.
        p_hat = m / n_shots_per_round
        # Use normal approx for the binomial proportion at the requested confidence.
        z = 1.96 if confidence > 0.9 else 1.0
        half_width = z * np.sqrt(p_hat * (1 - p_hat) / n_shots_per_round)
        p_lo_meas = max(0.0, p_hat - half_width)
        p_hi_meas = min(1.0, p_hat + half_width)
        # Invert via amplitude: p = sin²((2k+1)θ). Take the branch consistent
        # with the current [θ_lo, θ_hi].
        # For the simplest version, just narrow the interval around the
        # center of mass of the likelihood.
        # Center estimate of θ from p_hat:
        if (2 * k + 1) * (theta_hi - theta_lo) < np.pi:
            arg = np.clip(2 * p_hat - 1, -1, 1)
            theta_center = (np.arccos(arg) / (2 * (2 * k + 1)))
            # Use the branch near the midpoint of [theta_lo, theta_hi].
            possible = []
            for branch in range(2 * k + 2):
                cand = (branch * np.pi + (-1)**branch * np.arcsin(np.sqrt(p_hat))) / (2 * k + 1)
                if theta_lo - 1e-9 <= cand <= theta_hi + 1e-9:
                    possible.append(cand)
            if possible:
                theta_center = float(np.mean(possible))
                # Shrink interval around theta_center by half_width.
                # Use derivative dp/dθ = 2(2k+1) sin((2k+1)θ) cos((2k+1)θ)
                #                     = (2k+1) sin(2(2k+1)θ)
                deriv = abs((2 * k + 1) * np.sin(2 * (2 * k + 1) * theta_center))
                if deriv > 1e-9:
                    delta_theta = half_width / deriv
                    new_lo = max(theta_lo, theta_center - delta_theta)
                    new_hi = min(theta_hi, theta_center + delta_theta)
                    if new_hi > new_lo:
                        theta_lo, theta_hi = new_lo, new_hi

    theta_est = (theta_lo + theta_hi) / 2
    return {
        "theta_estimate":     float(theta_est),
        "amplitude_estimate": float(np.sin(theta_est) ** 2),
        "true_theta":         theta_true,
        "true_amplitude":     float(np.sin(theta_true) ** 2),
        "theta_interval":     (float(theta_lo), float(theta_hi)),
        "n_queries":          int(n_queries_total),
        "history":            history,
    }
