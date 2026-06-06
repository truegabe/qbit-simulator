"""Surrogate-gradient training for SNNs (Neftci et al. 2019).

Spike functions are non-differentiable: H(V - V_th) has zero gradient
everywhere except at threshold. Surrogate-gradient methods replace this
with a smooth pseudo-derivative during the backward pass, allowing
standard backprop-through-time on SNNs.

Common surrogates for sigma'(V - V_th):
  - "fast_sigmoid": 1 / (1 + |β(V-V_th)|)^2
  - "atan":        1 / (1 + (πβ(V-V_th))^2)
  - "gaussian":    exp(-(V-V_th)^2 / (2σ^2))

This module implements forward-mode SNN simulation (LIF with leaky
membrane), forward-pass spike emission, and computes gradients of a
loss over output spike-count via backward-through-time using the
surrogate derivative.

Implemented from scratch in numpy — no torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def fast_sigmoid_grad(x: np.ndarray, beta: float = 5.0) -> np.ndarray:
    return 1.0 / (1.0 + np.abs(beta * x)) ** 2


def atan_grad(x: np.ndarray, beta: float = 2.0) -> np.ndarray:
    return 1.0 / (1.0 + (np.pi * beta * x) ** 2)


@dataclass
class SurrogateGradSNN:
    """Single hidden-layer LIF SNN trained by BPTT with surrogate grads.

    Input → hidden LIF → output (linear sum of hidden spikes).
    """
    n_in: int
    n_hidden: int
    n_out: int
    tau: float = 5.0
    V_th: float = 1.0
    beta_surrogate: float = 5.0
    W_in: np.ndarray = field(default=None, repr=False)
    W_out: np.ndarray = field(default=None, repr=False)
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    def __post_init__(self) -> None:
        if self.W_in is None:
            self.W_in = self.rng.normal(0.0, 1.0 / np.sqrt(self.n_in),
                                          size=(self.n_hidden, self.n_in))
        if self.W_out is None:
            self.W_out = self.rng.normal(0.0, 1.0 / np.sqrt(self.n_hidden),
                                          size=(self.n_out, self.n_hidden))

    def forward(self, X: np.ndarray) -> dict:
        """X: shape (T, n_in) — input spike currents at each time step.

        Returns dict with V (T, n_hidden), S (T, n_hidden), y (n_out).
        """
        T = X.shape[0]
        decay = np.exp(-1.0 / self.tau)
        V = np.zeros((T, self.n_hidden))
        S = np.zeros((T, self.n_hidden))
        v = np.zeros(self.n_hidden)
        for t in range(T):
            v = decay * v + X[t] @ self.W_in.T
            s = (v >= self.V_th).astype(np.float64)
            v = v - self.V_th * s   # soft reset
            V[t] = v
            S[t] = s
        # Decode: average spike count weighted by W_out.
        y = self.W_out @ S.sum(axis=0)
        return {"V": V, "S": S, "y": y, "Vraw": V + self.V_th * S}

    def loss_and_grads(self, X: np.ndarray,
                        target: np.ndarray) -> tuple[float, dict]:
        """Squared-error loss on the spike-count readout."""
        out = self.forward(X)
        y = out["y"]; S = out["S"]; Vraw = out["Vraw"]
        loss = 0.5 * float(((y - target) ** 2).sum())
        # ∂loss/∂y = y - target.
        dy = y - target
        # ∂y/∂W_out = sum_t S_t.
        gW_out = np.outer(dy, S.sum(axis=0))
        # Backprop through time.
        T = X.shape[0]
        decay = np.exp(-1.0 / self.tau)
        # ∂loss/∂S_t = W_out^T @ dy  (sum over output dims).
        dS = np.tile((self.W_out.T @ dy)[None, :], (T, 1))
        # Walk back, accumulating ∂loss/∂v_t via surrogate derivative.
        gW_in = np.zeros_like(self.W_in)
        dv_next = np.zeros(self.n_hidden)
        for t in reversed(range(T)):
            # surrogate at the threshold: g'(V_raw - V_th).
            sg = fast_sigmoid_grad(Vraw[t] - self.V_th, beta=self.beta_surrogate)
            # dL/dv_t (after reset has been applied, before next step's decay).
            dv = dS[t] * sg + dv_next * decay * (1 - sg * self.V_th)
            # gW_in: ∂v_t / ∂W_in = X_t outer with identity in hidden dim.
            gW_in += np.outer(dv, X[t])
            dv_next = dv
        return loss, {"W_in": gW_in, "W_out": gW_out}

    def step_sgd(self, X: np.ndarray, target: np.ndarray,
                  lr: float = 0.01) -> float:
        loss, g = self.loss_and_grads(X, target)
        self.W_in -= lr * g["W_in"]
        self.W_out -= lr * g["W_out"]
        return loss
