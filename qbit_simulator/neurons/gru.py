"""Gated Recurrent Unit (Cho et al. 2014).

A simpler LSTM variant with two gates:

    z_t = σ(W_z · [x_t, h_{t-1}])         # update gate
    r_t = σ(W_r · [x_t, h_{t-1}])         # reset gate
    h~_t = tanh(W_h · [x_t, r_t ⊙ h_{t-1}])
    h_t  = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h~_t

Faster than LSTM, fewer parameters, often comparable performance.
This implementation uses simple BPTT.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


@dataclass
class GRUCell:
    n_in: int
    n_hidden: int
    n_out: int
    eta: float = 0.01
    clip: float = 5.0
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)
    W_z: np.ndarray = field(default=None, repr=False)
    W_r: np.ndarray = field(default=None, repr=False)
    W_h: np.ndarray = field(default=None, repr=False)
    b_z: np.ndarray = field(default=None, repr=False)
    b_r: np.ndarray = field(default=None, repr=False)
    b_h: np.ndarray = field(default=None, repr=False)
    W_hy: np.ndarray = field(default=None, repr=False)
    b_y:  np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        H = self.n_hidden; I = self.n_in
        std = 1.0 / np.sqrt(H + I)
        if self.W_z is None:
            self.W_z = self.rng.normal(0, std, (H, H + I))
            self.W_r = self.rng.normal(0, std, (H, H + I))
            self.W_h = self.rng.normal(0, std, (H, H + I))
            self.b_z = np.zeros(H); self.b_r = np.zeros(H); self.b_h = np.zeros(H)
            self.W_hy = self.rng.normal(0, std, (self.n_out, H))
            self.b_y  = np.zeros(self.n_out)

    def step(self, x: np.ndarray, h_prev: np.ndarray
              ) -> tuple[np.ndarray, dict]:
        zh = np.concatenate([h_prev, x])
        z = sigmoid(self.W_z @ zh + self.b_z)
        r = sigmoid(self.W_r @ zh + self.b_r)
        rh = np.concatenate([r * h_prev, x])
        h_tilde = np.tanh(self.W_h @ rh + self.b_h)
        h = (1 - z) * h_prev + z * h_tilde
        return h, {"zh": zh, "rh": rh, "z": z, "r": r, "h_tilde": h_tilde,
                    "h_prev": h_prev}

    def forward(self, X: np.ndarray) -> dict:
        T = X.shape[0]
        H = self.n_hidden
        h = np.zeros(H)
        Hs = np.zeros((T, H))
        Ys = np.zeros((T, self.n_out))
        caches = []
        for t in range(T):
            h, cache = self.step(X[t], h)
            Hs[t] = h
            Ys[t] = self.W_hy @ h + self.b_y
            caches.append(cache)
        return {"H": Hs, "Y": Ys, "caches": caches}

    def loss_and_step(self, X: np.ndarray, Y_target: np.ndarray) -> float:
        """Forward + simplified BPTT update via numerical-friendly chain.

        For brevity we use truncated BPTT with depth=T (full).
        """
        T = X.shape[0]
        H = self.n_hidden
        out = self.forward(X)
        Y = out["Y"]; Hs = out["H"]; caches = out["caches"]
        loss = 0.5 * float(((Y - Y_target) ** 2).sum())
        dY = Y - Y_target
        gW_hy = dY.T @ Hs
        gb_y  = dY.sum(axis=0)
        gWz = np.zeros_like(self.W_z); gWr = np.zeros_like(self.W_r)
        gWh = np.zeros_like(self.W_h)
        gbz = np.zeros_like(self.b_z); gbr = np.zeros_like(self.b_r)
        gbh = np.zeros_like(self.b_h)
        dh_next = np.zeros(H)
        for t in reversed(range(T)):
            cache = caches[t]
            dh = self.W_hy.T @ dY[t] + dh_next
            # h = (1-z) h_prev + z h~
            dz = dh * (cache["h_tilde"] - cache["h_prev"])
            dh_tilde = dh * cache["z"]
            dh_prev_direct = dh * (1 - cache["z"])
            # backprop through h_tilde = tanh(W_h rh + b_h)
            da_h = dh_tilde * (1 - cache["h_tilde"] ** 2)
            gWh += np.outer(da_h, cache["rh"])
            gbh += da_h
            drh = self.W_h.T @ da_h
            dr_h_prev = drh[:H]
            # drh splits into r*h_prev (so dr from r*h_prev) and x
            dr = dr_h_prev * cache["h_prev"]
            dh_prev_from_rh = dr_h_prev * cache["r"]
            # z gate
            da_z = dz * cache["z"] * (1 - cache["z"])
            gWz += np.outer(da_z, cache["zh"])
            gbz += da_z
            # r gate
            da_r = dr * cache["r"] * (1 - cache["r"])
            gWr += np.outer(da_r, cache["zh"])
            gbr += da_r
            # pass dh_next
            dzh_z = self.W_z.T @ da_z
            dzh_r = self.W_r.T @ da_r
            dh_next = dh_prev_direct + dh_prev_from_rh \
                      + dzh_z[:H] + dzh_r[:H]
        grads = {
            "W_z": gWz, "W_r": gWr, "W_h": gWh,
            "b_z": gbz, "b_r": gbr, "b_h": gbh,
            "W_hy": gW_hy, "b_y": gb_y,
        }
        for k in grads:
            np.clip(grads[k], -self.clip, self.clip, out=grads[k])
        for k, v in grads.items():
            setattr(self, k, getattr(self, k) - self.eta * v)
        return loss
