"""Edge-of-chaos diagnostic — Lyapunov spectrum health check.

Recurrent neural networks are most computationally useful at the
"edge of chaos": just below the transition from stable (ordered) to
chaotic dynamics. This regime gives long memory, rich transients, and
high computational capacity (Bertschinger & Natschläger 2004).

Diagnostic via the largest Lyapunov exponent λ_max:
    λ_max << 0  → strongly ordered, low memory, low capacity.
    λ_max  ≈ 0  → edge of chaos, productive regime.
    λ_max  > 0  → chaotic, no memory, high noise sensitivity.

For a rate network dx/dt = -x + W f(x), we measure how quickly an
infinitesimal perturbation grows or shrinks under the dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


def largest_lyapunov(W: np.ndarray,
                      activation: Callable[[np.ndarray], np.ndarray] = None,
                      n_steps: int = 500,
                      dt: float = 0.1,
                      tau: float = 1.0,
                      rng: np.random.Generator | None = None) -> float:
    """Estimate the largest Lyapunov exponent of dx/dt = (-x + W f(x))/tau.

    Algorithm: simulate x(t) and a perturbed twin y(t) = x + ε d̂.
    After dt, renormalize the perturbation, accumulating log-growth.
    """
    rng = rng or np.random.default_rng(0)
    if activation is None:
        activation = np.tanh
    n = W.shape[0]
    x = 0.1 * rng.standard_normal(n)
    # Warm-up.
    for _ in range(100):
        x = x + dt / tau * (-x + W @ activation(x))
    # Perturbation.
    d = rng.standard_normal(n); d /= np.linalg.norm(d)
    eps = 1e-8
    log_growth = 0.0
    for _ in range(n_steps):
        y = x + eps * d
        x_new = x + dt / tau * (-x + W @ activation(x))
        y_new = y + dt / tau * (-y + W @ activation(y))
        diff = y_new - x_new
        norm = np.linalg.norm(diff)
        if norm > 0:
            log_growth += np.log(norm / eps)
            d = diff / norm
        x = x_new
    return float(log_growth / (n_steps * dt))


def lyapunov_spectrum(W: np.ndarray,
                       activation: Callable[[np.ndarray], np.ndarray] = None,
                       n_exponents: int = 5,
                       n_steps: int = 500,
                       dt: float = 0.1,
                       tau: float = 1.0,
                       rng: np.random.Generator | None = None) -> np.ndarray:
    """Top-k Lyapunov exponents via Benettin's QR-based algorithm.

    Returns sorted (descending) array of n_exponents exponents.
    """
    rng = rng or np.random.default_rng(0)
    if activation is None:
        activation = np.tanh
    n = W.shape[0]
    x = 0.1 * rng.standard_normal(n)
    # Warm-up.
    for _ in range(100):
        x = x + dt / tau * (-x + W @ activation(x))
    # Initial orthonormal basis.
    Q = np.linalg.qr(rng.standard_normal((n, n_exponents)))[0]
    log_growth = np.zeros(n_exponents)
    for _ in range(n_steps):
        # Jacobian of f(x) = (-x + W tanh(x)) / tau at x:
        # J = (-I + W diag(1 - tanh²(x))) / tau
        if activation is np.tanh:
            f_prime = 1 - np.tanh(x) ** 2
        else:
            # Numerical derivative fallback.
            h = 1e-6
            f_prime = (activation(x + h) - activation(x - h)) / (2 * h)
        J = (-np.eye(n) + W * f_prime[None, :]) / tau
        # Discrete-time propagator: I + dt J.
        P = np.eye(n) + dt * J
        # Evolve x and basis.
        x = x + dt / tau * (-x + W @ activation(x))
        Q = P @ Q
        Q, R = np.linalg.qr(Q)
        log_growth += np.log(np.abs(np.diag(R)) + 1e-20)
    return log_growth / (n_steps * dt)


@dataclass
class EdgeOfChaosDiagnostic:
    """Lightweight health check for a recurrent network."""
    label_thresholds: tuple = (-0.05, 0.05)

    def diagnose(self, W: np.ndarray,
                  activation: Callable = None,
                  n_steps: int = 500,
                  rng: np.random.Generator | None = None) -> dict:
        """Returns dict with lyapunov, regime label, and recommendation."""
        lam = largest_lyapunov(W, activation=activation, n_steps=n_steps, rng=rng)
        low, high = self.label_thresholds
        if lam < low:
            regime = "ordered"; rec = "increase recurrent gain or reduce damping"
        elif lam > high:
            regime = "chaotic"; rec = "decrease recurrent gain or add inhibition"
        else:
            regime = "edge_of_chaos"; rec = "healthy — leave alone"
        return {"lyapunov_max": lam, "regime": regime, "recommendation": rec}
