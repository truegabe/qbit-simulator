"""Convolutional layers (CNN primitives).

V1-style visual processing is implemented as 2D convolutions.
This module gives:

  - Conv2D layer (cross-correlation, valid mode).
  - Max-pool 2D.
  - A small ConvNet for classification: conv → relu → pool → fc → softmax.

Numpy-only; trained by manual backprop on cross-entropy loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def conv2d_forward(X: np.ndarray, W: np.ndarray, b: np.ndarray
                    ) -> np.ndarray:
    """X: (C_in, H, W), W: (C_out, C_in, kh, kw), b: (C_out,)."""
    C_in, H, Wd = X.shape
    C_out, _, kh, kw = W.shape
    OH = H - kh + 1
    OW = Wd - kw + 1
    out = np.zeros((C_out, OH, OW))
    for o in range(C_out):
        for i in range(OH):
            for j in range(OW):
                patch = X[:, i:i + kh, j:j + kw]
                out[o, i, j] = (patch * W[o]).sum() + b[o]
    return out


def maxpool2d_forward(X: np.ndarray, k: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Returns pooled output and (argmax-index) array for backward pass."""
    C, H, W = X.shape
    OH = H // k; OW = W // k
    out = np.zeros((C, OH, OW))
    idx = np.zeros((C, OH, OW, 2), dtype=int)
    for c in range(C):
        for i in range(OH):
            for j in range(OW):
                patch = X[c, i*k:i*k+k, j*k:j*k+k]
                amax = np.argmax(patch)
                out[c, i, j] = patch.flat[amax]
                idx[c, i, j] = (amax // k, amax % k)
    return out, idx


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


@dataclass
class Conv2DLayer:
    """Single 2D convolutional layer with bias."""
    C_in: int
    C_out: int
    k: int = 3
    W: np.ndarray = field(default=None, repr=False)
    b: np.ndarray = field(default=None, repr=False)
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    def __post_init__(self) -> None:
        if self.W is None:
            fan_in = self.C_in * self.k * self.k
            self.W = self.rng.normal(0, np.sqrt(2.0 / fan_in),
                                       (self.C_out, self.C_in, self.k, self.k))
        if self.b is None:
            self.b = np.zeros(self.C_out)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return conv2d_forward(X, self.W, self.b)


@dataclass
class SimpleConvNet:
    """Conv + ReLU + max-pool + linear classifier on small images."""
    in_shape: tuple             # (C, H, W)
    n_classes: int
    n_filters: int = 4
    k: int = 3
    pool: int = 2
    eta: float = 0.01
    conv: Conv2DLayer = field(default=None)
    W_fc: np.ndarray = field(default=None, repr=False)
    b_fc: np.ndarray = field(default=None, repr=False)
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    def __post_init__(self) -> None:
        C, H, W = self.in_shape
        if self.conv is None:
            self.conv = Conv2DLayer(C_in=C, C_out=self.n_filters, k=self.k,
                                      rng=self.rng)
        oh = (H - self.k + 1) // self.pool
        ow = (W - self.k + 1) // self.pool
        self.flat_dim = self.n_filters * oh * ow
        self._oh = oh; self._ow = ow
        if self.W_fc is None:
            self.W_fc = self.rng.normal(0, np.sqrt(2.0 / self.flat_dim),
                                          (self.n_classes, self.flat_dim))
            self.b_fc = np.zeros(self.n_classes)

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict]:
        z_conv = self.conv.forward(X)
        z_relu = relu(z_conv)
        z_pool, pool_idx = maxpool2d_forward(z_relu, k=self.pool)
        flat = z_pool.reshape(-1)
        logits = self.W_fc @ flat + self.b_fc
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        return probs, {"X": X, "z_conv": z_conv, "z_relu": z_relu,
                        "z_pool": z_pool, "pool_idx": pool_idx, "flat": flat,
                        "probs": probs}

    def loss_and_step(self, X: np.ndarray, y: int) -> float:
        probs, cache = self.forward(X)
        loss = -np.log(probs[y] + 1e-9)
        # Backprop.
        d_logits = probs.copy(); d_logits[y] -= 1
        gW_fc = np.outer(d_logits, cache["flat"])
        gb_fc = d_logits
        d_flat = self.W_fc.T @ d_logits
        # Unflatten back to pool grid shape.
        d_pool = d_flat.reshape(self.n_filters, self._oh, self._ow)
        # Max-pool backward.
        d_relu = np.zeros_like(cache["z_relu"])
        for c in range(self.n_filters):
            for i in range(self._oh):
                for j in range(self._ow):
                    di, dj = cache["pool_idx"][c, i, j]
                    d_relu[c, i*self.pool + di, j*self.pool + dj] += d_pool[c, i, j]
        # ReLU backward.
        d_conv = d_relu * (cache["z_conv"] > 0)
        # Conv backward.
        gW_conv = np.zeros_like(self.conv.W)
        gb_conv = np.zeros_like(self.conv.b)
        C_in, H, Wd = X.shape
        OH, OW = d_conv.shape[1], d_conv.shape[2]
        for o in range(self.n_filters):
            gb_conv[o] = d_conv[o].sum()
            for i in range(OH):
                for j in range(OW):
                    gW_conv[o] += d_conv[o, i, j] * X[:, i:i + self.k, j:j + self.k]
        # Apply.
        self.W_fc -= self.eta * gW_fc
        self.b_fc -= self.eta * gb_fc
        self.conv.W -= self.eta * gW_conv
        self.conv.b -= self.eta * gb_conv
        return float(loss)

    def predict(self, X: np.ndarray) -> int:
        probs, _ = self.forward(X)
        return int(np.argmax(probs))
