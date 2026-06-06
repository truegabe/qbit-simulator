"""Multi-head self-attention + minimal Transformer encoder block.

Self-attention over a sequence:
    A = softmax(Q K^T / sqrt(d_k))
    out = A V

Multi-head: split Q, K, V into h heads, attend independently, concat.

This is a numpy-only inference + simple-loss implementation for studying
attention patterns. Trainable via finite differences or analytic
gradients (not implemented in full backprop here).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def softmax_2d(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                   mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    A = softmax_2d(scores, axis=-1)
    return A @ V, A


@dataclass
class MultiHeadAttention:
    """Multi-head self-attention layer."""
    d_model: int
    n_heads: int = 4
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)
    W_Q: np.ndarray = field(default=None, repr=False)
    W_K: np.ndarray = field(default=None, repr=False)
    W_V: np.ndarray = field(default=None, repr=False)
    W_O: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0
        std = 1.0 / np.sqrt(self.d_model)
        if self.W_Q is None:
            self.W_Q = self.rng.normal(0, std, (self.d_model, self.d_model))
            self.W_K = self.rng.normal(0, std, (self.d_model, self.d_model))
            self.W_V = self.rng.normal(0, std, (self.d_model, self.d_model))
            self.W_O = self.rng.normal(0, std, (self.d_model, self.d_model))

    def forward(self, X: np.ndarray, mask: np.ndarray | None = None) -> dict:
        """X: shape (T, d_model). Returns dict with output and attention."""
        T = X.shape[0]
        d_h = self.d_model // self.n_heads
        Q = (X @ self.W_Q.T).reshape(T, self.n_heads, d_h)
        K = (X @ self.W_K.T).reshape(T, self.n_heads, d_h)
        V = (X @ self.W_V.T).reshape(T, self.n_heads, d_h)
        outs = []
        attns = []
        for h in range(self.n_heads):
            out_h, A_h = scaled_dot_product_attention(Q[:, h], K[:, h], V[:, h], mask=mask)
            outs.append(out_h); attns.append(A_h)
        concat = np.concatenate(outs, axis=-1)
        output = concat @ self.W_O.T
        return {"output": output, "attention": np.stack(attns)}


@dataclass
class TransformerBlock:
    """Pre-norm transformer encoder block: attn → norm → FFN → norm."""
    d_model: int
    n_heads: int = 4
    d_ff: int = None
    attn: MultiHeadAttention = field(default=None)
    W_ff1: np.ndarray = field(default=None, repr=False)
    W_ff2: np.ndarray = field(default=None, repr=False)
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    def __post_init__(self) -> None:
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        if self.attn is None:
            self.attn = MultiHeadAttention(d_model=self.d_model,
                                             n_heads=self.n_heads, rng=self.rng)
        std = 1.0 / np.sqrt(self.d_model)
        if self.W_ff1 is None:
            self.W_ff1 = self.rng.normal(0, std, (self.d_ff, self.d_model))
            self.W_ff2 = self.rng.normal(0, std, (self.d_model, self.d_ff))

    def _layer_norm(self, x: np.ndarray) -> np.ndarray:
        m = x.mean(axis=-1, keepdims=True)
        s = x.std(axis=-1, keepdims=True) + 1e-6
        return (x - m) / s

    def forward(self, X: np.ndarray) -> np.ndarray:
        # Pre-norm attention.
        x_norm = self._layer_norm(X)
        attn_out = self.attn.forward(x_norm)["output"]
        x = X + attn_out
        # Pre-norm FFN.
        x_norm2 = self._layer_norm(x)
        h = np.maximum(x_norm2 @ self.W_ff1.T, 0)  # ReLU
        x = x + h @ self.W_ff2.T
        return x


def positional_encoding(T: int, d_model: int) -> np.ndarray:
    """Standard sinusoidal positional encoding."""
    pos = np.arange(T)[:, None]
    div = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    pe = np.zeros((T, d_model))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div[:pe[:, 1::2].shape[1]])
    return pe
