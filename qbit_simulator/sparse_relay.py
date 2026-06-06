"""Sparse relay -- top-k address-value encoding.

Only the k largest-magnitude activations are transmitted as
(index, value) pairs.  No codebook, no training, no projection matrix.
Works on any input; compression scales with sparsity.

Information-theoretic view
--------------------------
Dense float32 vector of n dims:  n * 32 bits
Sparse (index, value) pairs:     k * (ceil(log2(n)) + 32) bits

For n=1024, k=32, 10-bit index:  32 * 42 = 1344 bits vs 32768  -> 24x

Classes
-------
  SparseCode          -- lightweight container for (indices, values)
  SparseEncoder       -- dense activation -> SparseCode
  SparseDecoder       -- SparseCode -> dense activation
  SparseRelay         -- full encode -> transmit -> decode pipeline
  AdaptiveSparseRelay -- dynamically adjusts k based on activity budget
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# SparseCode
# ---------------------------------------------------------------------------

@dataclass
class SparseCode:
    """A sparse representation: parallel arrays of indices and values."""
    indices: np.ndarray   # (k,) int
    values:  np.ndarray   # (k,) float
    n_dims:  int          # original signal length

    @property
    def k(self) -> int:
        return len(self.indices)

    def density(self) -> float:
        return self.k / max(self.n_dims, 1)

    def bits(self, index_bits: Optional[int] = None) -> int:
        """Estimated bit cost of transmitting this code."""
        ib = index_bits or int(np.ceil(np.log2(max(self.n_dims, 2))))
        return self.k * (ib + 32)   # 32-bit float per value

    def dense(self, fill: float = 0.0) -> np.ndarray:
        """Reconstruct dense vector (same as decode with fill)."""
        x = np.full(self.n_dims, fill, dtype=np.float64)
        x[self.indices] = self.values
        return x


# ---------------------------------------------------------------------------
# SparseEncoder
# ---------------------------------------------------------------------------

class SparseEncoder:
    """Convert a dense activation to a SparseCode via top-k selection.

    Parameters
    ----------
    k          : number of elements to keep
    mode       : 'topk'      -- keep the k elements with largest |value|
                 'threshold' -- keep all elements with |value| > threshold
    threshold  : used in 'threshold' mode
    """

    def __init__(self, k: int = 16, mode: str = "topk",
                 threshold: float = 0.1) -> None:
        self.k         = k
        self.mode      = mode
        self.threshold = threshold

    def encode(self, x: np.ndarray) -> SparseCode:
        x = np.asarray(x, dtype=np.float64).ravel()
        n = len(x)

        if self.mode == "topk":
            k      = min(self.k, n)
            idx    = np.argpartition(np.abs(x), -k)[-k:]
            idx    = idx[np.argsort(np.abs(x[idx]))[::-1]]   # sorted by magnitude
        elif self.mode == "threshold":
            idx    = np.where(np.abs(x) > self.threshold)[0]
        else:
            raise ValueError(f"unknown mode: {self.mode}")

        return SparseCode(indices=idx.astype(int),
                          values=x[idx].copy(),
                          n_dims=n)

    def compression_ratio(self, x: np.ndarray) -> float:
        code  = self.encode(x)
        dense = len(x) * 32
        return dense / max(code.bits(), 1)

    def sparsity(self, x: np.ndarray) -> float:
        code = self.encode(x)
        return 1.0 - code.density()


# ---------------------------------------------------------------------------
# SparseDecoder
# ---------------------------------------------------------------------------

class SparseDecoder:
    """Reconstruct a dense vector from a SparseCode.

    Parameters
    ----------
    fill      : value for missing dimensions (default 0.0)
    smooth    : if True, apply gaussian blur to spread reconstructed
                values across ±1 neighbours (reduces quantisation artefacts)
    """

    def __init__(self, fill: float = 0.0, smooth: bool = False) -> None:
        self.fill   = fill
        self.smooth = smooth

    def decode(self, code: SparseCode) -> np.ndarray:
        x = np.full(code.n_dims, self.fill, dtype=np.float64)
        if len(code.indices) == 0:
            return x
        x[code.indices] = code.values
        if self.smooth and code.n_dims > 2:
            from scipy.ndimage import gaussian_filter1d   # optional
            x = gaussian_filter1d(x, sigma=0.5)
        return x


# ---------------------------------------------------------------------------
# SparseRelay
# ---------------------------------------------------------------------------

class SparseRelay:
    """Full encode -> (optional noise) -> decode pipeline.

    Parameters
    ----------
    n_dims      : signal dimensionality
    k           : number of active dimensions to transmit
    mode        : encoder mode ('topk' | 'threshold')
    threshold   : threshold for 'threshold' mode
    noise_std   : additive Gaussian noise on transmitted values (channel model)
    fill        : decoder fill value
    """

    def __init__(self, n_dims: int, k: int = 16,
                 mode: str = "topk", threshold: float = 0.1,
                 noise_std: float = 0.0, fill: float = 0.0) -> None:
        self.n_dims    = n_dims
        self.noise_std = noise_std
        self.encoder   = SparseEncoder(k=k, mode=mode, threshold=threshold)
        self.decoder   = SparseDecoder(fill=fill)
        self._total_sent: int = 0
        self._total_bits_dense: int = 0
        self._total_bits_sparse: int = 0

    # ------------------------------------------------------------------
    def transmit(self, x: np.ndarray,
                 rng: Optional[np.random.Generator] = None
                 ) -> tuple[np.ndarray, dict]:
        """Encode, simulate channel, decode.

        Returns
        -------
        x_rec  : reconstructed dense vector (n_dims,)
        stats  : dict with k_used, density, compression_ratio,
                 reconstruction_error, bits_sent
        """
        x    = np.asarray(x, dtype=np.float64).ravel()
        code = self.encoder.encode(x)

        # Channel noise on values.
        if self.noise_std > 0:
            rng   = rng or np.random.default_rng()
            noise = rng.standard_normal(code.k) * self.noise_std
            code  = SparseCode(indices=code.indices,
                                values=code.values + noise,
                                n_dims=code.n_dims)

        x_rec = self.decoder.decode(code)

        # Trim / pad to n_dims.
        x_rec = x_rec[:self.n_dims]
        if len(x_rec) < self.n_dims:
            x_rec = np.pad(x_rec, (0, self.n_dims - len(x_rec)))

        dense_bits  = self.n_dims * 32
        sparse_bits = code.bits()
        cr          = dense_bits / max(sparse_bits, 1)
        rec_err     = float(np.linalg.norm(x[:self.n_dims] - x_rec) /
                             (np.linalg.norm(x[:self.n_dims]) + 1e-12))

        self._total_sent       += 1
        self._total_bits_dense  += dense_bits
        self._total_bits_sparse += sparse_bits

        stats = {
            "k_used":               code.k,
            "density":              code.density(),
            "compression_ratio":    cr,
            "reconstruction_error": rec_err,
            "bits_sent":            sparse_bits,
            "bits_dense":           dense_bits,
        }
        return x_rec, stats

    def cumulative_stats(self) -> dict:
        return {
            "total_transmissions":  self._total_sent,
            "total_bits_dense":     self._total_bits_dense,
            "total_bits_sparse":    self._total_bits_sparse,
            "overall_compression":  (self._total_bits_dense /
                                     max(self._total_bits_sparse, 1)),
        }

    def __repr__(self) -> str:
        enc = self.encoder
        return (f"SparseRelay(n_dims={self.n_dims}, k={enc.k}, "
                f"mode='{enc.mode}', noise={self.noise_std})")


# ---------------------------------------------------------------------------
# AdaptiveSparseRelay
# ---------------------------------------------------------------------------

class AdaptiveSparseRelay:
    """SparseRelay that adjusts k to hit a target bit-rate budget.

    After each call the relay measures the actual compression ratio and
    tightens or loosens k (within [k_min, k_max]) to stay on budget.

    Parameters
    ----------
    n_dims      : signal dimensionality
    target_cr   : desired compression ratio (e.g. 10.0 = 10x compression)
    k_min, k_max: hard bounds on k
    lr          : adaptation step size (fraction of k range per step)
    """

    def __init__(self, n_dims: int, target_cr: float = 8.0,
                 k_min: int = 1, k_max: Optional[int] = None,
                 lr: float = 0.1, noise_std: float = 0.0) -> None:
        self.n_dims    = n_dims
        self.target_cr = target_cr
        self.k_min     = k_min
        self.k_max     = k_max or n_dims
        self.lr        = lr
        self._k        = max(k_min, n_dims // 8)
        self.noise_std = noise_std

    def _make_relay(self) -> SparseRelay:
        return SparseRelay(self.n_dims, k=self._k, noise_std=self.noise_std)

    def transmit(self, x: np.ndarray,
                 rng: Optional[np.random.Generator] = None
                 ) -> tuple[np.ndarray, dict]:
        relay        = self._make_relay()
        x_rec, stats = relay.transmit(x, rng)
        cr           = stats["compression_ratio"]
        # Adjust k: if cr > target compress more (reduce k), else increase k.
        delta = int(round(self.lr * (self.k_max - self.k_min)))
        delta = max(delta, 1)
        if cr > self.target_cr * 1.1:
            self._k = max(self.k_min, self._k - delta)
        elif cr < self.target_cr * 0.9:
            self._k = min(self.k_max, self._k + delta)
        stats["k_adapted"] = self._k
        return x_rec, stats

    @property
    def current_k(self) -> int:
        return self._k
