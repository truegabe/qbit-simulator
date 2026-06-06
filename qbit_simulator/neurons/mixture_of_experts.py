"""Mixture-of-Experts (Jacobs et al. 1991).

A gating network learns to route inputs to specialized expert
sub-networks. Each expert produces an output, and the gate produces a
softmax distribution over experts. Final output:

    y(x) = sum_e gate_e(x) · expert_e(x)

The biological inspiration: cortical regions / cortical columns
specialize for different feature classes, and "context" signals select
which region's output drives behavior.

This implementation: linear gating, linear experts, MSE loss,
trained by gradient descent on a 1D regression problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


@dataclass
class MixtureOfExperts:
    """Linear-gated mixture of linear experts for regression."""
    n_in: int
    n_out: int
    n_experts: int = 4
    eta: float = 0.01
    W_gate: np.ndarray = field(default=None, repr=False)
    W_exp:  np.ndarray = field(default=None, repr=False)
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    def __post_init__(self) -> None:
        if self.W_gate is None:
            self.W_gate = self.rng.normal(
                0.0, 0.1, size=(self.n_experts, self.n_in))
        if self.W_exp is None:
            self.W_exp = self.rng.normal(
                0.0, 0.1, size=(self.n_experts, self.n_out, self.n_in))

    def expert_outputs(self, x: np.ndarray) -> np.ndarray:
        return self.W_exp @ x   # shape (n_experts, n_out)

    def gate(self, x: np.ndarray) -> np.ndarray:
        return softmax(self.W_gate @ x)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        g = self.gate(x)
        e = self.expert_outputs(x)
        y = (g[:, None] * e).sum(axis=0)
        return y, g, e

    def step(self, x: np.ndarray, target: np.ndarray) -> float:
        y, g, e = self.forward(x)
        err = y - target                           # (n_out,)
        # Gradient w.r.t. expert weights: ∂L/∂W_exp_e = g_e · err ⊗ x.
        for k in range(self.n_experts):
            self.W_exp[k] -= self.eta * g[k] * np.outer(err, x)
        # Gradient w.r.t. gating logits z = W_gate x:
        # ∂L/∂z_k = err · (e_k - y) · g_k (using softmax derivative).
        s = np.array([float(err @ (e[k] - y)) for k in range(self.n_experts)]) * g
        self.W_gate -= self.eta * np.outer(s, x)
        return float(0.5 * (err ** 2).sum())

    def train(self, X: np.ndarray, Y: np.ndarray,
              n_epochs: int = 200) -> list:
        losses = []
        for ep in range(n_epochs):
            ep_loss = 0.0
            for i in self.rng.permutation(X.shape[0]):
                ep_loss += self.step(X[i], Y[i])
            losses.append(ep_loss / X.shape[0])
        return losses
