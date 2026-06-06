"""Optimal feedback control + LQR.

The Linear-Quadratic-Regulator finds the optimal control policy for a
linear dynamical system with quadratic cost:

    dx/dt = A x + B u
    cost  = ∫ (x^T Q x + u^T R u) dt

The solution is u = -K x where K is computed via the algebraic
Riccati equation:

    A^T P + P A - P B R^{-1} B^T P + Q = 0
    K = R^{-1} B^T P
"""

from __future__ import annotations

import numpy as np


def solve_riccati(A: np.ndarray, B: np.ndarray, Q: np.ndarray,
                   R: np.ndarray, n_iter: int = 1000,
                   tol: float = 1e-9) -> np.ndarray:
    """Iterative solution of continuous algebraic Riccati equation."""
    P = Q.copy()
    R_inv = np.linalg.inv(R)
    for _ in range(n_iter):
        BRB = B @ R_inv @ B.T
        P_new = P + 0.01 * (A.T @ P + P @ A - P @ BRB @ P + Q)
        if np.max(np.abs(P_new - P)) < tol:
            P = P_new
            break
        P = P_new
    return P


def lqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray,
         R: np.ndarray) -> np.ndarray:
    """Return optimal feedback gain K (u = -K x)."""
    P = solve_riccati(A, B, Q, R)
    R_inv = np.linalg.inv(R)
    return R_inv @ B.T @ P


def simulate_lqr(A: np.ndarray, B: np.ndarray, K: np.ndarray,
                  x0: np.ndarray, n_steps: int = 200,
                  dt: float = 0.05) -> np.ndarray:
    x = x0.copy()
    traj = np.zeros((n_steps, len(x0)))
    for t in range(n_steps):
        u = -K @ x
        x = x + dt * (A @ x + B @ u)
        traj[t] = x
    return traj
