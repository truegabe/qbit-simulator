"""Quantum-aware gradient methods and optimizers for VQE-style algorithms.

Provided here:
    - parameter_shift_gradient: exact analytic gradient via the parameter-
      shift rule. Each gradient component requires two circuit evaluations
      at θ ± π/2. This is the canonical method used on real quantum hardware
      (no finite-difference noise, no auto-diff overhead).
    - quantum_natural_gradient: gradient descent in the Fubini-Study metric.
      Reuses the McLachlan M-matrix from VarQITE. Often dramatically faster
      convergence than vanilla gradient descent for VQE problems.
    - adam: classical Adam optimizer (epsilon-stabilized RMSProp + momentum)
      for use with parameter-shift gradients.

These are drop-in alternatives to scipy's minimize for parameterized-quantum-
circuit optimization. They expose the same `cost, grad → step` interface
that's used everywhere else in this simulator.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def parameter_shift_gradient(
    cost_fn: Callable[[np.ndarray], float],
    theta: np.ndarray,
    shift: float = np.pi / 2,
) -> np.ndarray:
    """Compute ∇C(θ) via the parameter-shift rule.

    For each parameter i: ∂C/∂θ_i = (C(θ + s·e_i) - C(θ - s·e_i)) / (2 sin(s))
    With s = π/2, this becomes exact for gates whose Hermitian generator
    has two eigenvalues ±1 (e.g. Rx, Ry, Rz on a single qubit).
    """
    n = len(theta)
    grad = np.zeros(n, dtype=np.float64)
    denom = 2 * np.sin(shift)
    for i in range(n):
        e_i = np.zeros(n); e_i[i] = shift
        c_plus  = cost_fn(theta + e_i)
        c_minus = cost_fn(theta - e_i)
        grad[i] = (c_plus - c_minus) / denom
    return grad


def adam(
    cost_fn: Callable[[np.ndarray], float],
    theta0: np.ndarray,
    grad_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    lr: float = 0.05,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    n_iter: int = 200,
    tol: float = 1e-6,
) -> dict:
    """Adam optimizer with parameter-shift gradients by default.

    Args:
        cost_fn:  callable θ → cost.
        theta0:   initial parameter vector.
        grad_fn:  callable θ → ∇cost(θ). Defaults to parameter-shift.
        lr:       learning rate.
        beta1, beta2, eps: standard Adam hyperparameters.
        n_iter:   maximum iterations.
        tol:      stop when |cost change| < tol for 5 iterations.

    Returns:
        dict with theta_opt, cost_opt, history, n_iters.
    """
    if grad_fn is None:
        grad_fn = lambda t: parameter_shift_gradient(cost_fn, t)
    theta = np.asarray(theta0, dtype=np.float64).copy()
    n = len(theta)
    m = np.zeros(n); v = np.zeros(n)
    history: list[float] = []
    stall_count = 0
    prev_cost = float("inf")
    for t in range(1, n_iter + 1):
        c = float(cost_fn(theta))
        history.append(c)
        if abs(prev_cost - c) < tol:
            stall_count += 1
            if stall_count >= 5:
                break
        else:
            stall_count = 0
        prev_cost = c
        g = grad_fn(theta)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        theta -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return {
        "theta_opt": theta,
        "cost_opt":  float(cost_fn(theta)),
        "history":   history,
        "n_iters":   len(history),
    }


def quantum_natural_gradient(
    cost_fn: Callable[[np.ndarray], float],
    state_fn: Callable[[np.ndarray], np.ndarray],
    theta0: np.ndarray,
    lr: float = 0.1,
    n_iter: int = 100,
    regularization: float = 1e-6,
    eps_grad: float = 1e-4,
) -> dict:
    """Gradient descent in the Fubini-Study metric.

    Update rule: θ ← θ - lr · M⁻¹ · ∇C
    where M_ij = Re ⟨∂_i ψ | ∂_j ψ⟩ (Fubini-Study metric).

    Args:
        cost_fn:   callable θ → cost.
        state_fn:  callable θ → state vector (for computing M).
        theta0:    initial parameter vector.
        lr:        learning rate.
        n_iter:    iteration count.
        regularization: ridge term on M for numerical stability.
        eps_grad:  finite-difference epsilon.
    """
    theta = np.asarray(theta0, dtype=np.float64).copy()
    n_p = len(theta)
    history: list[float] = []
    for _ in range(n_iter):
        c = float(cost_fn(theta))
        history.append(c)

        # Build M and ∇C with finite differences.
        psi = state_fn(theta)
        grads_psi = []
        for i in range(n_p):
            e_i = np.zeros(n_p); e_i[i] = eps_grad
            psi_p = state_fn(theta + e_i)
            psi_m = state_fn(theta - e_i)
            grads_psi.append((psi_p - psi_m) / (2 * eps_grad))
        M = np.zeros((n_p, n_p), dtype=np.float64)
        for i in range(n_p):
            for j in range(n_p):
                M[i, j] = float(np.real(np.vdot(grads_psi[i], grads_psi[j])))

        # ∇C via finite differences.
        grad_c = np.zeros(n_p)
        for i in range(n_p):
            e_i = np.zeros(n_p); e_i[i] = eps_grad
            grad_c[i] = (cost_fn(theta + e_i) - cost_fn(theta - e_i)) / (2 * eps_grad)

        M_reg = M + regularization * np.eye(n_p)
        try:
            step = np.linalg.solve(M_reg, grad_c)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(M_reg, grad_c, rcond=None)[0]
        theta -= lr * step

    return {
        "theta_opt": theta,
        "cost_opt":  float(cost_fn(theta)),
        "history":   history,
    }
