"""Deep predictive coding network (Rao & Ballard 1999; Lotter et al.).

A hierarchical generative model with multiple layers. Each layer
maintains a "representation" r_l and a top-down "prediction" of the
layer below; the bottom-up error signals get propagated up.

Per-layer dynamics:
  prediction        : p_l = W_l r_{l+1}
  prediction error  : e_l = r_l - p_l
  update            : r_{l+1} += η · W_l^T e_l - λ · r_{l+1}

Weight update (Hebbian on error * representation):
  W_l += η_W · e_l r_{l+1}^T

The bottom layer's r_0 is clamped to the input.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DeepPredictiveCoder:
    """Stacked predictive coding network."""
    layer_sizes: list             # [n_input, n_h1, n_h2, ...]
    eta_r: float = 0.1
    eta_W: float = 0.005
    lam: float = 0.01
    n_inference_steps: int = 20
    Ws: list = field(default_factory=list)
    rs: list = field(default_factory=list)
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    def __post_init__(self) -> None:
        if not self.Ws:
            for l in range(len(self.layer_sizes) - 1):
                n_below = self.layer_sizes[l]
                n_above = self.layer_sizes[l + 1]
                self.Ws.append(self.rng.normal(
                    0, 1.0 / np.sqrt(n_above), size=(n_below, n_above)))
        if not self.rs:
            for l in range(len(self.layer_sizes)):
                self.rs.append(np.zeros(self.layer_sizes[l]))

    def infer(self, x: np.ndarray) -> None:
        """Clamp r_0 = x and iteratively settle higher representations."""
        self.rs[0] = x.copy()
        for _ in range(self.n_inference_steps):
            for l in range(1, len(self.rs)):
                p_below = self.Ws[l - 1] @ self.rs[l]
                e_below = self.rs[l - 1] - p_below
                # Bottom-up gradient.
                grad = self.Ws[l - 1].T @ e_below - self.lam * self.rs[l]
                self.rs[l] += self.eta_r * grad

    def update_weights(self) -> None:
        for l in range(len(self.Ws)):
            p_below = self.Ws[l] @ self.rs[l + 1]
            e_below = self.rs[l] - p_below
            self.Ws[l] += self.eta_W * np.outer(e_below, self.rs[l + 1])

    def train(self, X: np.ndarray, n_iter: int = 500) -> list:
        losses = []
        for it in range(n_iter):
            x = X[self.rng.integers(0, X.shape[0])]
            self.infer(x)
            # Loss = sum of squared errors across layers.
            loss = 0.0
            for l in range(len(self.Ws)):
                p = self.Ws[l] @ self.rs[l + 1]
                loss += 0.5 * ((self.rs[l] - p) ** 2).sum()
            losses.append(loss)
            self.update_weights()
        return losses

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        self.infer(x)
        return self.Ws[0] @ self.rs[1]
