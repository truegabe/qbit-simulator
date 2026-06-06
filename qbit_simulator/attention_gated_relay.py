"""Attention-gated relay -- query/key dynamic channel selection.

Instead of always transmitting a fixed set of channels (as in SparseRelay),
the attention gate lets the RECEIVER specify a query vector that controls
which channels are forwarded.

Biological analogy
------------------
Top-down attention from prefrontal cortex sends a query ("looking for edge
detectors") to V1, which responds by boosting channels that match the query
and suppressing others.  Only the attended channels cross the relay.

Mechanism
---------
  Q  (query)  : context vector describing what the receiver wants
  K  (keys)   : one key vector per channel of the input
  A  (scores) : A_i = softmax(Q . K_i / sqrt(d_k))
  Select top-k channels by attention score, transmit those.

The key matrix K is learned (Hebbian: associate each channel with a context
that tends to activate it).  Alternatively K can be fixed (random or identity).

Classes
-------
  AttentionGate          -- computes scores, selects top-k channels
  KeyMatrix              -- learnable K with Hebbian update
  AttentionGatedRelay    -- full encode -> route -> decode pipeline
  MultiHeadAttentionGate -- h independent attention heads, outputs merged
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# KeyMatrix
# ---------------------------------------------------------------------------

class KeyMatrix:
    """Per-channel key vectors K  (n_channels x d_key).

    Each row K[i] represents what context query would activate channel i.

    Hebbian update:
        K[i] <- (1-lr) * K[i]  +  lr * query   when channel i is active.
    """

    def __init__(self, n_channels: int, d_key: int,
                 init: str = "random", seed: int = 0,
                 lr: float = 0.01) -> None:
        self.n_channels = n_channels
        self.d_key      = d_key
        self.lr         = lr
        rng             = np.random.default_rng(seed)

        if init == "random":
            raw      = rng.standard_normal((n_channels, d_key))
            norms    = np.linalg.norm(raw, axis=1, keepdims=True)
            self.K   = raw / np.maximum(norms, 1e-12)
        elif init == "identity":
            # Use first d_key dims as key (truncate / pad).
            self.K = np.eye(n_channels, d_key)
        elif init == "zeros":
            self.K = np.zeros((n_channels, d_key))
        else:
            raise ValueError(f"unknown init: {init}")

    def update(self, active_indices: np.ndarray,
               query: np.ndarray) -> None:
        """Hebbian update for the channels that were selected."""
        q = np.asarray(query, dtype=np.float64).ravel()
        if len(q) < self.d_key:
            q = np.pad(q, (0, self.d_key - len(q)))
        else:
            q = q[:self.d_key]
        for i in active_indices:
            self.K[i] = (1 - self.lr) * self.K[i] + self.lr * q
            norm       = np.linalg.norm(self.K[i])
            if norm > 1e-12:
                self.K[i] /= norm

    def scores(self, query: np.ndarray) -> np.ndarray:
        """Dot-product attention scores for all channels.  (n_channels,)"""
        q = np.asarray(query, dtype=np.float64).ravel()
        if len(q) < self.d_key:
            q = np.pad(q, (0, self.d_key - len(q)))
        else:
            q = q[:self.d_key]
        scale = np.sqrt(max(self.d_key, 1))
        return (self.K @ q) / scale


# ---------------------------------------------------------------------------
# AttentionGate
# ---------------------------------------------------------------------------

class AttentionGate:
    """Select top-k channels of x based on attention to a query.

    Parameters
    ----------
    n_channels  : total input channels
    d_key       : key/query dimensionality
    k           : number of channels to pass through
    key_init    : key matrix initialisation ('random' | 'identity' | 'zeros')
    hebbian_lr  : Hebbian learning rate for key update
    temperature : softmax temperature (lower = sharper)
    seed        : reproducibility
    """

    def __init__(self, n_channels: int, d_key: int, k: int = 16,
                 key_init: str = "random", hebbian_lr: float = 0.01,
                 temperature: float = 1.0, seed: int = 0) -> None:
        self.n_channels  = n_channels
        self.d_key       = d_key
        self.k           = min(k, n_channels)
        self.temperature = temperature
        self.keys        = KeyMatrix(n_channels, d_key, init=key_init,
                                      seed=seed, lr=hebbian_lr)

    def attend(self, x: np.ndarray,
               query: np.ndarray,
               learn: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply attention gate to input x given query.

        Returns
        -------
        x_gated   : (n_channels,) -- x with non-attended dims zeroed
        top_idx   : (k,) int -- selected channel indices
        weights   : (k,) float -- attention weights for selected channels
        """
        x     = np.asarray(x, dtype=np.float64).ravel()
        query = np.asarray(query, dtype=np.float64).ravel()

        raw_scores = self.keys.scores(query) / max(self.temperature, 1e-6)

        # Softmax over all channels.
        raw_scores = raw_scores - raw_scores.max()   # numerical stability
        weights_all = np.exp(raw_scores)
        weights_all /= weights_all.sum()

        # Top-k by weight.
        top_idx = np.argpartition(weights_all, -self.k)[-self.k:]
        top_idx = top_idx[np.argsort(weights_all[top_idx])[::-1]]

        # Gate: pass only top-k channels.
        # Normalise weights over the selected subset so they sum to 1,
        # then scale back to unit gain (preserves signal energy).
        x_gated   = np.zeros_like(x)
        sel_w     = weights_all[top_idx]
        sel_w_sum = sel_w.sum()
        norm_w    = sel_w / max(sel_w_sum, 1e-12)
        for idx, w in zip(top_idx, norm_w):
            x_gated[idx] = x[idx] * w * self.k   # re-scale to ~unit gain

        if learn:
            self.keys.update(top_idx, query)

        return x_gated, top_idx, weights_all[top_idx]

    def __repr__(self) -> str:
        return (f"AttentionGate(n_channels={self.n_channels}, "
                f"d_key={self.d_key}, k={self.k})")


# ---------------------------------------------------------------------------
# AttentionGatedRelay
# ---------------------------------------------------------------------------

class AttentionGatedRelay:
    """Full attention-gated relay pipeline.

    The relay accepts a context query along with each activation vector.
    Only the top-k most query-relevant channels are transmitted.

    Parameters
    ----------
    n_dims      : input/output signal dimensionality
    d_key       : key/query vector size
    k           : channels to pass per transmission
    key_init    : key matrix init strategy
    hebbian_lr  : Hebbian learning rate
    temperature : attention temperature
    noise_std   : additive channel noise on transmitted values
    seed        : reproducibility
    """

    def __init__(self, n_dims: int, d_key: int = 32, k: int = 16,
                 key_init: str = "random", hebbian_lr: float = 0.01,
                 temperature: float = 1.0, noise_std: float = 0.0,
                 seed: int = 0) -> None:
        self.n_dims    = n_dims
        self.noise_std = noise_std
        self.gate      = AttentionGate(n_dims, d_key, k=k, key_init=key_init,
                                        hebbian_lr=hebbian_lr,
                                        temperature=temperature, seed=seed)
        self._n_calls: int   = 0
        self._err_acc: float = 0.0

    # ------------------------------------------------------------------
    def transmit(self, x: np.ndarray, query: np.ndarray,
                 rng: Optional[np.random.Generator] = None,
                 learn: bool = True) -> tuple[np.ndarray, dict]:
        """Gate x by query, transmit selected channels, reconstruct.

        Returns
        -------
        x_rec  : reconstructed signal (n_dims,) -- non-selected dims = 0
        stats  : dict with k_used, top_indices, attention_weights,
                 compression_ratio, reconstruction_error
        """
        x     = np.asarray(x, dtype=np.float64).ravel()
        query = np.asarray(query, dtype=np.float64).ravel()

        x_gated, top_idx, weights = self.gate.attend(x, query, learn=learn)

        if self.noise_std > 0:
            rng = rng or np.random.default_rng()
            x_gated[top_idx] += rng.standard_normal(len(top_idx)) * self.noise_std

        # Reconstruct: gated signal is the reconstruction (non-attended = 0).
        x_rec   = x_gated

        k        = self.gate.k
        cr       = self.n_dims / max(k, 1)
        rec_err  = float(np.linalg.norm(x - x_rec) /
                          (np.linalg.norm(x) + 1e-12))

        self._n_calls += 1
        self._err_acc += rec_err

        stats = {
            "k_used":               k,
            "top_indices":          top_idx.tolist(),
            "attention_weights":    weights.tolist(),
            "compression_ratio":    cr,
            "reconstruction_error": rec_err,
            "mean_error_so_far":    self._err_acc / self._n_calls,
        }
        return x_rec, stats

    def __repr__(self) -> str:
        g = self.gate
        return (f"AttentionGatedRelay(n_dims={self.n_dims}, "
                f"d_key={g.d_key}, k={g.k})")


# ---------------------------------------------------------------------------
# MultiHeadAttentionGate
# ---------------------------------------------------------------------------

class MultiHeadAttentionGate:
    """h independent attention heads; outputs are merged by max-pooling.

    Each head has its own KeyMatrix and selects k channels independently.
    The union of selected channels is transmitted (up to h*k channels total).

    Parameters
    ----------
    n_channels : input channels
    d_key      : key dimensionality per head
    n_heads    : number of attention heads
    k_per_head : channels selected per head
    """

    def __init__(self, n_channels: int, d_key: int = 32,
                 n_heads: int = 4, k_per_head: int = 8,
                 temperature: float = 1.0, seed: int = 0) -> None:
        self.n_channels = n_channels
        self.n_heads    = n_heads
        self.k_per_head = k_per_head
        self.heads      = [
            AttentionGate(n_channels, d_key, k=k_per_head,
                           temperature=temperature, seed=seed + h)
            for h in range(n_heads)
        ]

    def attend(self, x: np.ndarray,
               queries: np.ndarray,
               learn: bool = True) -> tuple[np.ndarray, dict]:
        """Apply all heads and merge outputs.

        Parameters
        ----------
        queries : (n_heads, d_key) or (d_key,) array.
                  If 1-D, the same query is broadcast to all heads.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        Q = np.asarray(queries, dtype=np.float64)
        if Q.ndim == 1:
            Q = np.tile(Q, (self.n_heads, 1))

        x_merged  = np.zeros_like(x)
        all_idx   = set()
        for h, head in enumerate(self.heads):
            q              = Q[h] if h < len(Q) else Q[-1]
            x_gated, idx, w = head.attend(x, q, learn=learn)
            # Max-pool: keep the largest activation across heads.
            mask            = np.abs(x_gated) > np.abs(x_merged)
            x_merged[mask]  = x_gated[mask]
            all_idx.update(idx.tolist())

        k_total = len(all_idx)
        stats   = {
            "k_total":   k_total,
            "all_idx":   sorted(all_idx),
            "n_heads":   self.n_heads,
            "density":   k_total / max(self.n_channels, 1),
        }
        return x_merged, stats
