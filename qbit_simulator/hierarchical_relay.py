"""Hierarchical relay -- multi-hop compression pipeline.

Models the cortical hierarchy: V1 -> V2 -> V4 -> IT -> PFC.
Each hop compresses the signal further; the reverse path reconstructs.

Biological grounding
--------------------
- Bottom-up:  each area sends a compressed abstract representation upward.
- Top-down:   each area sends prediction errors or attention signals downward.
- The hierarchy creates a trade-off: more compression = more robustness
  to noise, but higher reconstruction error at the leaves.
- Each hub can optionally apply a non-linear activation (like a cortical
  area computing a non-linear feature map).

Implementation
--------------
  HierarchyLevel  -- one node in the hierarchy (encoder + decoder matrix)
  HierarchicalRelay -- chain: input -> L1 -> L2 -> ... -> Lk -> output
                       with full up-pass (compression) and down-pass (reconstruction)

Example
-------
  relay = HierarchicalRelay(dims=[256, 64, 16, 4])
  compressed, up_stats = relay.encode(x)     # 256 -> 64 -> 16 -> 4
  x_rec, down_stats    = relay.decode(compressed, target_dim=256)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np


# ---------------------------------------------------------------------------
# HierarchyLevel
# ---------------------------------------------------------------------------

class HierarchyLevel:
    """Single level in the hierarchy: encoder (W_down) and decoder (W_up).

    W_down : (n_out, n_in)   -- compresses n_in -> n_out
    W_up   : (n_in,  n_out)  -- reconstructs n_out -> n_in

    Optionally a nonlinearity is applied after the encode step.

    Parameters
    ----------
    n_in      : input dimensionality (from the level below)
    n_out     : output dimensionality (to the level above)
    name      : label (e.g. 'V1->V2')
    init      : weight initialisation ('random' | 'pca')
    nonlin    : nonlinearity applied after encode (None | 'relu' | 'tanh')
    noise_std : noise added to compressed representation
    seed      : for reproducibility
    """

    def __init__(self, n_in: int, n_out: int, name: str = "",
                 init: str = "random", nonlin: Optional[str] = "relu",
                 noise_std: float = 0.0, seed: int = 0) -> None:
        self.n_in      = n_in
        self.n_out     = n_out
        self.name      = name or f"{n_in}->{n_out}"
        self.noise_std = noise_std
        self.nonlin    = nonlin

        rng = np.random.default_rng(seed)
        if init == "random":
            W          = rng.standard_normal((n_out, n_in))
            # Normalise rows so each output has unit expected norm.
            W         /= np.maximum(
                np.linalg.norm(W, axis=1, keepdims=True), 1e-12)
            self.W_down = W
            # Pseudo-inverse via W^T (W W^T)^{-1}: for unit-row W, W W^T ~ I,
            # so W^T is a good approximation scaled by 1/n_out.
            WWT        = W @ W.T
            WWT_inv    = np.linalg.pinv(WWT)
            self.W_up  = W.T @ WWT_inv      # (n_in, n_out)

        elif init == "pca":
            # Random orthogonal basis (Gram-Schmidt on random matrix).
            k          = min(n_out, n_in)
            raw        = rng.standard_normal((n_in, k))
            Q, _       = np.linalg.qr(raw)   # Q: (n_in, k), orthonormal cols
            self.W_down = Q[:, :k].T          # (k, n_in) -- compresses n_in -> k
            self.W_up   = Q[:, :k]            # (n_in, k) -- exact left-inverse
            # If n_out > k, pad with zeros to reach requested output size.
            if n_out > k:
                pad          = np.zeros((n_out - k, n_in))
                self.W_down  = np.vstack([self.W_down, pad])   # (n_out, n_in)
                self.W_up    = np.hstack(
                    [self.W_up, np.zeros((n_in, n_out - k))])  # (n_in, n_out)
        else:
            raise ValueError(f"unknown init: {init}")

    # ------------------------------------------------------------------
    def encode(self, x: np.ndarray,
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Bottom-up pass: compress x from n_in to n_out."""
        x = np.asarray(x, dtype=np.float64).ravel()
        h = self.W_down @ x
        if self.nonlin == "relu":
            h = np.maximum(h, 0.0)
        elif self.nonlin == "tanh":
            h = np.tanh(h)
        if self.noise_std > 0:
            rng = rng or np.random.default_rng()
            h  += rng.standard_normal(len(h)) * self.noise_std
        return h

    def decode(self, h: np.ndarray) -> np.ndarray:
        """Top-down pass: reconstruct x from n_out to n_in."""
        h = np.asarray(h, dtype=np.float64).ravel()
        return self.W_up @ h

    @property
    def compression_ratio(self) -> float:
        return self.n_in / max(self.n_out, 1)

    def __repr__(self) -> str:
        return f"HierarchyLevel({self.name}, cr={self.compression_ratio:.1f}x)"


# ---------------------------------------------------------------------------
# HierarchicalRelay
# ---------------------------------------------------------------------------

class HierarchicalRelay:
    """Multi-hop compression pipeline: input -> L1 -> L2 -> ... -> top.

    Parameters
    ----------
    dims      : list of dimensionalities [n_input, n_L1, n_L2, ..., n_top].
                Example: [256, 64, 16, 4] creates 3 levels.
    init      : weight initialisation for all levels
    nonlin    : nonlinearity for all levels (None | 'relu' | 'tanh')
    noise_std : noise per level
    seed      : base seed (each level gets seed + level_index)
    names     : optional list of level names (length = len(dims) - 1)
    """

    def __init__(self, dims: list[int],
                 init: str = "random", nonlin: Optional[str] = "relu",
                 noise_std: float = 0.0, seed: int = 0,
                 names: Optional[list[str]] = None) -> None:
        if len(dims) < 2:
            raise ValueError("dims must have at least 2 entries (input + 1 level)")
        self.dims   = dims
        self.levels = []
        for i in range(len(dims) - 1):
            nm = (names[i] if names and i < len(names)
                  else f"L{i}({dims[i]}->{dims[i+1]})")
            self.levels.append(
                HierarchyLevel(dims[i], dims[i + 1], name=nm,
                                init=init, nonlin=nonlin,
                                noise_std=noise_std, seed=seed + i))

        self._n_encode: int    = 0
        self._n_decode: int    = 0
        self._err_acc:  float  = 0.0

    # ------------------------------------------------------------------
    @property
    def n_levels(self) -> int:
        return len(self.levels)

    @property
    def total_compression(self) -> float:
        cr = 1.0
        for lv in self.levels:
            cr *= lv.compression_ratio
        return cr

    # ------------------------------------------------------------------
    def encode(self, x: np.ndarray,
               rng: Optional[np.random.Generator] = None
               ) -> tuple[np.ndarray, dict]:
        """Bottom-up pass through all levels.

        Returns
        -------
        h_top    : compressed representation at the top level (dims[-1],)
        stats    : dict with per-level shapes, total compression
        """
        x   = np.asarray(x, dtype=np.float64).ravel()
        h   = x
        per = []
        for lv in self.levels:
            h_new = lv.encode(h, rng)
            per.append({"name": lv.name, "in": len(h), "out": len(h_new),
                        "cr": lv.compression_ratio})
            h = h_new

        self._n_encode += 1
        stats = {
            "levels":            per,
            "total_compression": self.total_compression,
            "top_dim":           len(h),
            "input_dim":         len(x),
        }
        return h, stats

    def decode(self, h: np.ndarray,
               target_dim: Optional[int] = None) -> tuple[np.ndarray, dict]:
        """Top-down pass: reconstruct from top representation.

        Parameters
        ----------
        h          : top-level representation
        target_dim : expected output dimension (default = dims[0])

        Returns
        -------
        x_rec  : reconstructed signal
        stats  : dict with per-level reconstruction info
        """
        h   = np.asarray(h, dtype=np.float64).ravel()
        per = []
        for lv in reversed(self.levels):
            h_new = lv.decode(h)
            per.append({"name": lv.name, "in": len(h), "out": len(h_new)})
            h = h_new

        if target_dim is not None and len(h) != target_dim:
            if len(h) > target_dim:
                h = h[:target_dim]
            else:
                h = np.pad(h, (0, target_dim - len(h)))

        self._n_decode += 1
        stats = {"levels": per, "output_dim": len(h)}
        return h, stats

    def relay(self, x: np.ndarray,
              rng: Optional[np.random.Generator] = None
              ) -> tuple[np.ndarray, dict]:
        """Full encode -> decode round-trip.

        Returns
        -------
        x_rec  : reconstructed input
        stats  : dict with total_compression, reconstruction_error, per-level info
        """
        x        = np.asarray(x, dtype=np.float64).ravel()
        h, up    = self.encode(x, rng)
        x_rec, dn = self.decode(h, target_dim=len(x))

        rec_err = float(np.linalg.norm(x - x_rec) /
                         (np.linalg.norm(x) + 1e-12))
        self._err_acc += rec_err

        stats = {
            "total_compression":    self.total_compression,
            "reconstruction_error": rec_err,
            "mean_error_so_far":    self._err_acc / max(self._n_decode, 1),
            "top_dim":              up["top_dim"],
            "up_levels":            up["levels"],
            "down_levels":          dn["levels"],
        }
        return x_rec, stats

    # ------------------------------------------------------------------
    def intermediate_representations(self, x: np.ndarray
                                     ) -> list[np.ndarray]:
        """Return activations at every level (including input).

        Useful for visualisation / analysis.
        """
        x    = np.asarray(x, dtype=np.float64).ravel()
        reps = [x.copy()]
        h    = x
        for lv in self.levels:
            h = lv.encode(h)
            reps.append(h.copy())
        return reps

    def __repr__(self) -> str:
        s = " -> ".join(str(d) for d in self.dims)
        return f"HierarchicalRelay([{s}], cr={self.total_compression:.1f}x)"


# ---------------------------------------------------------------------------
# AdaptiveHierarchicalRelay
# ---------------------------------------------------------------------------

class AdaptiveHierarchicalRelay(HierarchicalRelay):
    """HierarchicalRelay that can prune levels under a bit-rate constraint.

    After each relay call the reconstruction error is monitored.
    If error < target the relay uses fewer levels (more compression);
    if error > target it uses more levels (less compression).

    Parameters
    ----------
    dims         : full hierarchy dimensions
    target_error : reconstruction error target (0.1 = 10%)
    adapt_rate   : how quickly to change active levels
    """

    def __init__(self, dims: list[int], target_error: float = 0.1,
                 adapt_rate: float = 0.1, **kwargs) -> None:
        super().__init__(dims, **kwargs)
        self.target_error  = target_error
        self.adapt_rate    = adapt_rate
        self._active_depth = len(self.levels)   # start with all levels
        self._ema_err      = 0.0

    def relay(self, x: np.ndarray,
              rng: Optional[np.random.Generator] = None
              ) -> tuple[np.ndarray, dict]:
        x  = np.asarray(x, dtype=np.float64).ravel()
        # Use only _active_depth levels.
        orig_levels = self.levels
        self.levels = orig_levels[:self._active_depth]

        x_rec, stats = super().relay(x, rng)

        self.levels = orig_levels

        # Update EMA of reconstruction error and adapt depth.
        err = stats["reconstruction_error"]
        self._ema_err = 0.9 * self._ema_err + 0.1 * err
        if self._ema_err < self.target_error * 0.8 and self._active_depth > 1:
            self._active_depth -= 1
        elif self._ema_err > self.target_error * 1.2:
            self._active_depth = min(self._active_depth + 1, len(orig_levels))

        stats["active_depth"]    = self._active_depth
        stats["ema_error"]       = self._ema_err
        return x_rec, stats
