"""Attention mechanism: soft gating over connectivity.

In neuroscience, "attention" refers to the dynamic ENHANCEMENT of
certain neural signals over others — modulating effective gain on a
moment-to-moment basis. Computationally it's equivalent to a soft
multiplicative gate:

    output_i  =  sum_j  α_ij(context) · w_ij · x_j

where α_ij ∈ [0, 1] is the attention weight, modulated by some
"query" / "context" signal. The transformer attention mechanism is one
particular instantiation; biological attention shares the structure.

This module provides:

  - `softmax_attention(query, keys, values)`: vanilla scaled dot-product
    attention from the Transformer paper, used for sanity checks.
  - `BiologicalAttention(n)`: attention as a multiplicative gain field
    over a neural population. Gain is set by a "saliency map" computed
    bottom-up + top-down.
  - `WinnerTakeAll`: a strong-competition variant where only the
    most-active neuron fires.
  - `compute_saliency_map(features, top_down)`: combine bottom-up
    feature contrast with top-down task bias.

Provides both the math (helper functions) and a stateful gain-modulator
object (`AttentionGate`) suitable for inserting between SNN layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------------
# Softmax (transformer-style) attention
# ----------------------------------------------------------------------------

def softmax_attention(
    query: np.ndarray, keys: np.ndarray, values: np.ndarray,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Scaled dot-product attention.

    Args:
        query:  shape (d_k,).
        keys:   shape (N, d_k).
        values: shape (N, d_v).
        temperature: τ (higher = softer attention).

    Returns:
        (output, attention_weights):
          output shape (d_v,), weights shape (N,).
    """
    d_k = query.shape[0]
    scores = (keys @ query) / np.sqrt(d_k) / max(temperature, 1e-6)
    scores = scores - scores.max()        # numerical stability
    weights = np.exp(scores)
    weights = weights / weights.sum()
    out = weights @ values
    return out, weights


# ----------------------------------------------------------------------------
# Saliency map (bottom-up + top-down)
# ----------------------------------------------------------------------------

def compute_saliency_map(
    features: np.ndarray, top_down: np.ndarray | None = None,
    bu_weight: float = 1.0, td_weight: float = 1.0,
) -> np.ndarray:
    """Bottom-up + top-down saliency.

    Bottom-up: high contrast features (deviating from local average)
    are salient. We compute |x − mean(x)|.

    Top-down: task-relevant features (specified externally).

    Args:
        features:  shape (n,) — feature activations.
        top_down:  shape (n,) — task-bias map (optional, defaults to 0).
        bu_weight, td_weight: relative contributions.

    Returns:
        saliency map shape (n,), values in [0, 1].
    """
    n = len(features)
    bu = np.abs(features - features.mean())
    if bu.max() > 1e-9:
        bu = bu / bu.max()
    if top_down is None:
        td = np.zeros(n)
    else:
        td = np.asarray(top_down, dtype=np.float64)
        if td.shape != (n,):
            raise ValueError(f"top_down must have shape ({n},)")
    combined = bu_weight * bu + td_weight * td
    if combined.max() > 1e-9:
        combined = combined / combined.max()
    return combined


# ----------------------------------------------------------------------------
# Biological attention gate
# ----------------------------------------------------------------------------

@dataclass
class AttentionGate:
    """A multiplicative gain field that modulates n incoming signals.

    The gain g_i ∈ [g_min, g_max] is set by the saliency map. Apply to
    a signal vector via `gate.apply(signal)`.
    """
    n:        int
    g_min:    float = 0.1     # baseline (suppressed) gain
    g_max:    float = 2.0     # peak (attended) gain
    saliency: np.ndarray = field(default=None, repr=False)
    gain:     np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.saliency is None:
            self.saliency = np.zeros(self.n)
        if self.gain is None:
            self.gain = np.full(self.n, (self.g_min + self.g_max) / 2)

    def update_from_saliency(self, saliency: np.ndarray) -> None:
        """Compute gain from saliency: g = g_min + (g_max − g_min)·saliency."""
        if saliency.shape != (self.n,):
            raise ValueError(f"saliency shape {saliency.shape} != ({self.n},)")
        self.saliency = saliency
        self.gain = self.g_min + (self.g_max - self.g_min) * saliency

    def apply(self, signal: np.ndarray) -> np.ndarray:
        """Multiplicative gain modulation."""
        if signal.shape != (self.n,):
            raise ValueError(f"signal shape {signal.shape} != ({self.n},)")
        return signal * self.gain


# ----------------------------------------------------------------------------
# Winner-take-all (extreme attention)
# ----------------------------------------------------------------------------

def winner_take_all(signal: np.ndarray, k: int = 1) -> np.ndarray:
    """Keep the top-k values, zero out the rest.

    k=1 → strict winner-take-all.
    """
    out = np.zeros_like(signal)
    if k >= len(signal):
        return signal.copy()
    indices = np.argpartition(signal, -k)[-k:]
    out[indices] = signal[indices]
    return out


def soft_winner_take_all(signal: np.ndarray, temperature: float = 0.1
                           ) -> np.ndarray:
    """Softmax-based winner-take-all: returns a normalized distribution
    sharpened by `temperature` (lower = harder competition)."""
    s = signal / max(temperature, 1e-6)
    s = s - s.max()
    out = np.exp(s)
    return out / out.sum()


# ----------------------------------------------------------------------------
# Multi-head attention (very small)
# ----------------------------------------------------------------------------

def multi_head_attention(
    query: np.ndarray, keys: np.ndarray, values: np.ndarray,
    n_heads: int = 2, temperature: float = 1.0,
) -> np.ndarray:
    """Split features into n_heads, run softmax_attention on each,
    concatenate. Returns aggregated output."""
    d_k = query.shape[0]
    if d_k % n_heads != 0:
        raise ValueError(f"d_k {d_k} not divisible by n_heads {n_heads}")
    head_dim = d_k // n_heads
    outputs = []
    for h in range(n_heads):
        sl = slice(h * head_dim, (h + 1) * head_dim)
        q_h = query[sl]
        k_h = keys[:, sl]
        v_h = values  # we don't split values here for simplicity
        o_h, _ = softmax_attention(q_h, k_h, v_h, temperature=temperature)
        outputs.append(o_h)
    return np.mean(outputs, axis=0)
