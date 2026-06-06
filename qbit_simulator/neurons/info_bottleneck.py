"""Information Bottleneck (Tishby et al. 1999).

Find a stochastic mapping p(T | X) such that the representation T:
  - retains maximal info about Y: I(T; Y) large.
  - compresses X: I(T; X) small.

Tradeoff: maximize  I(T; Y) - β · I(T; X).

We implement the classical IB algorithm on a discrete joint p(X, Y):
iterative updates of p(t | x), p(t), p(y | t) — converges to a local
optimum.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Row-wise KL(p || q) over the last axis."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((p > 0) & (q > 0), np.log(p / q), 0.0)
    return (p * ratio).sum(axis=-1)


@dataclass
class InformationBottleneck:
    n_clusters: int = 4
    beta: float = 1.0
    max_iter: int = 100
    tol: float = 1e-5
    p_t_given_x: np.ndarray = field(default=None, repr=False)
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    def fit(self, p_xy: np.ndarray) -> None:
        """p_xy: shape (n_x, n_y) — joint distribution."""
        n_x, n_y = p_xy.shape
        T = self.n_clusters
        p_x = p_xy.sum(axis=1)
        p_y_given_x = p_xy / (p_x[:, None] + 1e-12)
        # Initialize p(t | x) randomly.
        p_t_x = self.rng.uniform(size=(n_x, T))
        p_t_x /= p_t_x.sum(axis=1, keepdims=True)
        prev_obj = -np.inf
        for it in range(self.max_iter):
            # p(t).
            p_t = p_x @ p_t_x        # (T,)
            # p(y | t) = sum_x p(t | x) p(x) p(y | x) / p(t).
            p_yt = (p_t_x * p_x[:, None]).T @ p_y_given_x   # (T, n_y)
            p_y_given_t = p_yt / (p_t[:, None] + 1e-12)
            # Update p(t | x) using Boltzmann form.
            # d(y|x, t) = KL(p(y|x) || p(y|t))
            d = np.zeros((n_x, T))
            for tt in range(T):
                d[:, tt] = kl_divergence(p_y_given_x, p_y_given_t[tt])
            log_p_t_x = np.log(p_t + 1e-12) - self.beta * d
            log_p_t_x -= log_p_t_x.max(axis=1, keepdims=True)
            p_t_x = np.exp(log_p_t_x)
            p_t_x /= p_t_x.sum(axis=1, keepdims=True)
            # Objective: I(T;Y) - beta * I(T;X) (approx).
            I_TX = (p_t_x * p_x[:, None] * np.log(
                (p_t_x + 1e-12) / (p_t[None, :] + 1e-12))).sum()
            I_TY = 0.0
            for tt in range(T):
                if p_t[tt] > 1e-12:
                    I_TY += p_t[tt] * (p_y_given_t[tt] * np.log(
                        (p_y_given_t[tt] + 1e-12) / (p_xy.sum(axis=0) + 1e-12))).sum()
            obj = I_TY - self.beta * I_TX
            if abs(obj - prev_obj) < self.tol:
                break
            prev_obj = obj
        self.p_t_given_x = p_t_x

    def transform(self, x: int) -> np.ndarray:
        return self.p_t_given_x[x]
