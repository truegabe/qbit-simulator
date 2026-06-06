"""FORCE learning (Sussillo & Abbott 2009).

A reservoir of rate neurons has fixed random recurrent weights J.
A linear readout w produces output z = w^T r. FORCE trains w online
using Recursive Least Squares to drive z toward a target signal f(t).

The key trick: feedback the output BACK into the reservoir while
training. This keeps the reservoir close to the regime where the
readout works, even at the start when w is random.

Update (RLS):
    P <- P - (P r r^T P) / (1 + r^T P r)
    e = z - f
    w <- w - e · P r
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FORCE:
    n: int = 200
    p_conn: float = 0.1
    g: float = 1.5
    tau: float = 10.0
    alpha: float = 1.0    # RLS regularization
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)
    J: np.ndarray = field(default=None, repr=False)
    J_GF: np.ndarray = field(default=None, repr=False)
    w: np.ndarray = field(default=None, repr=False)
    P: np.ndarray = field(default=None, repr=False)
    x: np.ndarray = field(default=None, repr=False)
    r: np.ndarray = field(default=None, repr=False)
    z: float = 0.0

    def __post_init__(self) -> None:
        n = self.n
        if self.J is None:
            mask = self.rng.uniform(size=(n, n)) < self.p_conn
            self.J = self.g * self.rng.normal(0, 1/np.sqrt(self.p_conn * n),
                                                (n, n)) * mask
            np.fill_diagonal(self.J, 0)
        if self.J_GF is None:
            self.J_GF = self.rng.uniform(-1, 1, n)
        if self.w is None:
            self.w = np.zeros(n)
        if self.P is None:
            self.P = np.eye(n) / self.alpha
        if self.x is None:
            self.x = 0.5 * self.rng.standard_normal(n)
        if self.r is None:
            self.r = np.tanh(self.x)

    def step(self, dt: float = 1.0, target: float | None = None) -> float:
        """One simulation step. If target is given, apply RLS update."""
        dx = (-self.x + self.J @ self.r + self.J_GF * self.z) / self.tau
        self.x += dt * dx
        self.r = np.tanh(self.x)
        self.z = float(self.w @ self.r)
        if target is not None:
            Pr = self.P @ self.r
            denom = 1.0 + float(self.r @ Pr)
            self.P -= np.outer(Pr, Pr) / denom
            e = self.z - target
            self.w -= e * (self.P @ self.r)
        return self.z

    def run(self, target_fn, n_steps: int = 1000,
             train: bool = True, dt: float = 1.0) -> np.ndarray:
        zs = np.zeros(n_steps)
        for t in range(n_steps):
            tgt = target_fn(t) if (train and callable(target_fn)) else None
            zs[t] = self.step(dt=dt, target=tgt)
        return zs
