"""Long Short-Term Memory (LSTM) cell + sequence trainer.

Forget / input / output gates control a cell state c_t that flows
through time with multiplicative gating, avoiding vanishing gradients:

    f_t = σ(W_f · [x_t, h_{t-1}] + b_f)
    i_t = σ(W_i · [x_t, h_{t-1}] + b_i)
    g_t = tanh(W_g · [x_t, h_{t-1}] + b_g)
    o_t = σ(W_o · [x_t, h_{t-1}] + b_o)
    c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
    h_t = o_t ⊙ tanh(c_t)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


@dataclass
class LSTMCell:
    n_in: int
    n_hidden: int
    n_out: int
    eta: float = 0.01
    clip: float = 5.0
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)
    W: np.ndarray = field(default=None, repr=False)   # (4 H, H+I)
    b: np.ndarray = field(default=None, repr=False)
    W_hy: np.ndarray = field(default=None, repr=False)
    b_y:  np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        H = self.n_hidden
        I = self.n_in
        std = 1.0 / np.sqrt(H + I)
        if self.W is None:
            self.W = self.rng.normal(0, std, (4 * H, H + I))
            self.b = np.zeros(4 * H)
            # Initialize forget gate bias to 1 (helps learning).
            self.b[:H] = 1.0
            self.W_hy = self.rng.normal(0, std, (self.n_out, H))
            self.b_y  = np.zeros(self.n_out)

    def step(self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray, dict]:
        H = self.n_hidden
        z = np.concatenate([h_prev, x])
        a = self.W @ z + self.b
        f = sigmoid(a[:H])
        i = sigmoid(a[H:2*H])
        g = np.tanh(a[2*H:3*H])
        o = sigmoid(a[3*H:])
        c = f * c_prev + i * g
        h = o * np.tanh(c)
        return h, c, {"z": z, "a": a, "f": f, "i": i, "g": g, "o": o, "c": c}

    def forward(self, X: np.ndarray) -> dict:
        T = X.shape[0]
        H = self.n_hidden
        h = np.zeros(H); c = np.zeros(H)
        Hs = np.zeros((T, H)); Cs = np.zeros((T, H))
        Ys = np.zeros((T, self.n_out))
        caches = []
        for t in range(T):
            h, c, cache = self.step(X[t], h, c)
            Hs[t] = h; Cs[t] = c
            Ys[t] = self.W_hy @ h + self.b_y
            caches.append(cache)
        return {"H": Hs, "C": Cs, "Y": Ys, "caches": caches}

    def loss_and_grads(self, X: np.ndarray, Y_target: np.ndarray
                        ) -> tuple[float, dict]:
        T = X.shape[0]
        H = self.n_hidden
        out = self.forward(X)
        Y = out["Y"]; Hs = out["H"]; Cs = out["C"]; caches = out["caches"]
        loss = 0.5 * float(((Y - Y_target) ** 2).sum())
        dY = Y - Y_target
        gW_hy = dY.T @ Hs
        gb_y  = dY.sum(axis=0)
        gW = np.zeros_like(self.W); gb = np.zeros_like(self.b)
        dh_next = np.zeros(H); dc_next = np.zeros(H)
        for t in reversed(range(T)):
            dh = self.W_hy.T @ dY[t] + dh_next
            cache = caches[t]
            tanh_c = np.tanh(cache["c"])
            do = dh * tanh_c
            dc = dh * cache["o"] * (1 - tanh_c ** 2) + dc_next
            df = dc * (Cs[t - 1] if t > 0 else np.zeros(H))
            di = dc * cache["g"]
            dg = dc * cache["i"]
            # Backprop through activations.
            da_f = df * cache["f"] * (1 - cache["f"])
            da_i = di * cache["i"] * (1 - cache["i"])
            da_g = dg * (1 - cache["g"] ** 2)
            da_o = do * cache["o"] * (1 - cache["o"])
            da = np.concatenate([da_f, da_i, da_g, da_o])
            gW += np.outer(da, cache["z"])
            gb += da
            dz = self.W.T @ da
            dh_next = dz[:H]
            dc_next = dc * cache["f"]
        grads = {"W": gW, "b": gb, "W_hy": gW_hy, "b_y": gb_y}
        for k in grads:
            np.clip(grads[k], -self.clip, self.clip, out=grads[k])
        return loss, grads

    def step_sgd(self, X: np.ndarray, Y_target: np.ndarray) -> float:
        loss, g = self.loss_and_grads(X, Y_target)
        self.W   -= self.eta * g["W"]
        self.b   -= self.eta * g["b"]
        self.W_hy -= self.eta * g["W_hy"]
        self.b_y  -= self.eta * g["b_y"]
        return loss
