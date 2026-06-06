"""Higher-order Trotter-Suzuki decompositions for Hamiltonian simulation.

For H = sum_k H_k (sum of non-commuting terms), the Lie-Trotter formula

    exp(-i H t) ≈ [prod_k exp(-i H_k t/N)]^N

is FIRST-ORDER accurate: error scales as O(t² / N). Going to higher
order requires Suzuki recursion (Suzuki 1991):

  * Order 2: S_2(t) = exp(-i H_1 t/2) … exp(-i H_M t/2) ·
                       exp(-i H_M t/2) … exp(-i H_1 t/2).
             Error: O(t³).
  * Order 4: S_4(t) = S_2(s·t)² · S_2((1-4s)·t) · S_2(s·t)²
             with s = 1/(4 - 4^(1/3)).
             Error: O(t⁵).
  * Order 2k (k ≥ 2): recursive formula.

For very long evolutions or tight error tolerances, higher orders
dramatically reduce the number of Trotter steps required.

This module provides:

  - `trotter_step_order1(H_list, t)`: forward Lie-Trotter step.
  - `trotter_step_order2(H_list, t)`: symmetric (second-order).
  - `trotter_step_order4(H_list, t)`: fourth-order Suzuki.
  - `trotter_step_order_2k(H_list, t, k)`: general order 2k.
  - `evolve_order(H_list, t, n_steps, order)`: full evolution.
  - `trotter_error(H_list, t, n_steps, order)`: difference from exact.

We work with dense matrix Hamiltonians; trade-off between order and
step cost depends on system size.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.linalg import expm


# ----------------------------------------------------------------------------
# Trotter steps of various orders
# ----------------------------------------------------------------------------

def trotter_step_order1(H_list: Sequence[np.ndarray], t: float) -> np.ndarray:
    """First-order: prod_k exp(-i H_k t)."""
    d = H_list[0].shape[0]
    U = np.eye(d, dtype=np.complex128)
    for H in H_list:
        U = expm(-1j * H * t) @ U
    return U


def trotter_step_order2(H_list: Sequence[np.ndarray], t: float) -> np.ndarray:
    """Symmetric (Strang) second-order:

        S_2(t) = exp(-i H_1 t/2) … exp(-i H_M t/2)
                · exp(-i H_M t/2) … exp(-i H_1 t/2)
    """
    d = H_list[0].shape[0]
    U = np.eye(d, dtype=np.complex128)
    for H in H_list:
        U = expm(-1j * H * t / 2) @ U
    for H in reversed(H_list):
        U = expm(-1j * H * t / 2) @ U
    return U


def trotter_step_order4(H_list: Sequence[np.ndarray], t: float) -> np.ndarray:
    """Fourth-order Suzuki:

        S_4(t) = S_2(s·t)² · S_2((1 − 4s)·t) · S_2(s·t)²
        s = 1 / (4 − 4^(1/3))
    """
    s = 1.0 / (4 - 4 ** (1 / 3))
    sub = trotter_step_order2(H_list, s * t)
    middle = trotter_step_order2(H_list, (1 - 4 * s) * t)
    return sub @ sub @ middle @ sub @ sub


def trotter_step_order_2k(H_list: Sequence[np.ndarray], t: float,
                            k: int) -> np.ndarray:
    """General order 2k Suzuki (k ≥ 1).

    Recursion: S_{2k}(t) = S_{2k-2}(u_k t)² · S_{2k-2}((1-4u_k) t) · S_{2k-2}(u_k t)²
    with u_k = 1 / (4 − 4^(1/(2k-1))).
    """
    if k == 1:
        return trotter_step_order2(H_list, t)
    if k < 1:
        raise ValueError("k must be ≥ 1")
    u_k = 1.0 / (4 - 4 ** (1.0 / (2 * k - 1)))
    sub = trotter_step_order_2k(H_list, u_k * t, k - 1)
    middle = trotter_step_order_2k(H_list, (1 - 4 * u_k) * t, k - 1)
    return sub @ sub @ middle @ sub @ sub


# ----------------------------------------------------------------------------
# Full evolution
# ----------------------------------------------------------------------------

def evolve_order(
    H_list: Sequence[np.ndarray],
    t: float, n_steps: int = 1,
    order: int = 2,
) -> np.ndarray:
    """Trotter-evolved unitary at total time `t` in `n_steps` steps."""
    dt = t / n_steps
    if order == 1:
        step_fn = trotter_step_order1
    elif order == 2:
        step_fn = trotter_step_order2
    elif order == 4:
        step_fn = trotter_step_order4
    else:
        if order % 2 != 0:
            raise ValueError("order must be even (or 1)")
        k = order // 2
        step_fn = lambda H, t: trotter_step_order_2k(H, t, k)
    d = H_list[0].shape[0]
    U = np.eye(d, dtype=np.complex128)
    U_step = step_fn(H_list, dt)
    for _ in range(n_steps):
        U = U_step @ U
    return U


def trotter_error(
    H_list: Sequence[np.ndarray],
    t: float, n_steps: int = 1, order: int = 2,
) -> float:
    """Operator-norm distance between the Trotter approximation and the
    exact exp(-i H t).
    """
    H_total = sum(H_list)
    U_exact = expm(-1j * H_total * t)
    U_trot = evolve_order(H_list, t, n_steps, order)
    return float(np.linalg.norm(U_exact - U_trot, ord=2))


# ----------------------------------------------------------------------------
# Asymptotic scaling check
# ----------------------------------------------------------------------------

def error_scaling_table(
    H_list: Sequence[np.ndarray],
    t: float, order: int,
    n_steps_list: list[int],
) -> list[tuple[int, float]]:
    """Compute (n_steps, error) for a range of step counts."""
    return [(N, trotter_error(H_list, t, N, order)) for N in n_steps_list]
