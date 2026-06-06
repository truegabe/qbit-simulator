"""Vanilla recurrent neural network with BPTT.

The simplest recurrent network:

    h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)
    y_t = W_hy h_t + b_y

Trained on sequences (X, Y) of shape (T, n_in) and (T, n_out) by
back-propagation through time with optional gradient clipping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class VanillaRNN:
    n_in: int
    n_hidden: int
    n_out: int
    eta: float = 0.01
    clip: float = 5.0
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)
    W_xh: np.ndarray = field(default=None, repr=False)
    W_hh: np.ndarray = field(default=None, repr=False)
    W_hy: np.ndarray = field(default=None, repr=False)
    b_h:  np.ndarray = field(default=None, repr=False)
    b_y:  np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        std_in = 1.0 / np.sqrt(self.n_in)
        std_h  = 1.0 / np.sqrt(self.n_hidden)
        if self.W_xh is None:
            self.W_xh = self.rng.normal(0, std_in, (self.n_hidden, self.n_in))
            self.W_hh = self.rng.normal(0, std_h,  (self.n_hidden, self.n_hidden))
            self.W_hy = self.rng.normal(0, std_h,  (self.n_out, self.n_hidden))
            self.b_h  = np.zeros(self.n_hidden)
            self.b_y  = np.zeros(self.n_out)

    def forward(self, X: np.ndarray, h0: np.ndarray | None = None) -> dict:
        """X: (T, n_in). Returns dict with H (T, n_hidden) and Y (T, n_out)."""
        T = X.shape[0]
        h = np.zeros(self.n_hidden) if h0 is None else h0.copy()
        H = np.zeros((T, self.n_hidden))
        Y = np.zeros((T, self.n_out))
        for t in range(T):
            h = np.tanh(self.W_xh @ X[t] + self.W_hh @ h + self.b_h)
            H[t] = h
            Y[t] = self.W_hy @ h + self.b_y
        return {"H": H, "Y": Y, "h_last": h}

    def loss_and_grads(self, X: np.ndarray, Y_target: np.ndarray,
                        h0: np.ndarray | None = None) -> tuple[float, dict]:
        T = X.shape[0]
        out = self.forward(X, h0=h0)
        H = out["H"]; Y = out["Y"]
        # MSE loss.
        loss = 0.5 * float(((Y - Y_target) ** 2).sum())
        dY = Y - Y_target            # (T, n_out)
        gW_hy = dY.T @ H
        gb_y  = dY.sum(axis=0)
        # BPTT.
        gW_xh = np.zeros_like(self.W_xh)
        gW_hh = np.zeros_like(self.W_hh)
        gb_h  = np.zeros_like(self.b_h)
        dh_next = np.zeros(self.n_hidden)
        for t in reversed(range(T)):
            dh = self.W_hy.T @ dY[t] + dh_next
            dh_raw = (1 - H[t] ** 2) * dh        # tanh derivative
            gb_h  += dh_raw
            gW_xh += np.outer(dh_raw, X[t])
            h_prev = H[t - 1] if t > 0 else np.zeros(self.n_hidden)
            gW_hh += np.outer(dh_raw, h_prev)
            dh_next = self.W_hh.T @ dh_raw
        grads = {"W_xh": gW_xh, "W_hh": gW_hh, "W_hy": gW_hy,
                  "b_h": gb_h, "b_y": gb_y}
        # Clip.
        for k in grads:
            np.clip(grads[k], -self.clip, self.clip, out=grads[k])
        return loss, grads

    def step_sgd(self, X: np.ndarray, Y_target: np.ndarray,
                  lr: float | None = None) -> float:
        if lr is None:
            lr = self.eta
        loss, g = self.loss_and_grads(X, Y_target)
        self.W_xh -= lr * g["W_xh"]
        self.W_hh -= lr * g["W_hh"]
        self.W_hy -= lr * g["W_hy"]
        self.b_h  -= lr * g["b_h"]
        self.b_y  -= lr * g["b_y"]
        return loss
