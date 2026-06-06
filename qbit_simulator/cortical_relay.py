"""Cortical relay -- compressed qudit communication bus.

Sits between brain modules (neurons/) and carries compressed symbolic
codes instead of full activation vectors.  This is the white-matter
analogue: it does not think, it transports efficiently.

Architecture
------------

    [Region A]  -- any brain module that outputs a numpy vector
        |
    ActivationEncoder
        |   maps N-dim float vector -> k qudit symbols (each 0..d-1)
        |   compression ratio: N / (k * log2(d))
        v
    QuditChannel
        |   transmits k qudit symbols (perfect or noisy)
        |   latency model: k * symbol_delay   vs   N * spike_delay (raw)
        v
    ActivationDecoder
        |   maps k qudit symbols -> N-dim float vector (reconstruction)
        v
    [Region B]  -- any brain module that accepts a numpy vector

The codebook is learned from data (fit() method) or can be set manually.
Under the hood:

  Encoding:
    1. Project activation x (N-dim) onto k basis vectors  -> k floats
    2. Quantize each float to one of d=10 levels (0..9)   -> k integers
    3. Optionally entangle the k qudit symbols via CSUM gates

  Decoding:
    1. Map each integer back to a float (centre of its quantization bin)
    2. Reconstruct via the transpose of the projection basis

The speed argument
------------------
  Raw channel:        latency ~ N  * spike_delay          (e.g. 48 spikes)
  Relay channel:      latency ~ k  * qudit_symbol_delay   (e.g. 4 symbols)

  For k=4, d=10, N=48: 12x fewer channel uses.
  Encoding/decoding is local compute (parallel in a real neural tissue).

Classes
-------
  QuditCodebook       -- learn/store the projection + quantization mapping
  QuditChannel        -- simulate the qudit channel (latency + noise)
  CorticalRelay       -- full encode -> channel -> decode pipeline
  RelayProfiler       -- tracks bandwidth, latency, reconstruction error
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .qudit import QuditCircuit, X_power, H_gate


# ---------------------------------------------------------------------------
# QuditCodebook  (projection + vector quantization)
# ---------------------------------------------------------------------------

@dataclass
class QuditCodebook:
    """Learns a mapping: R^N  <->  {0..d-1}^k

    Parameters
    ----------
    n_dims  : dimensionality of the activation vector (N)
    n_symbols : number of qudit symbols in the code (k)
    d       : qudit dimension (default 10)
    rng     : numpy random generator
    """
    n_dims:    int
    n_symbols: int
    d:         int  = 10
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    # Learned attributes (set by fit or manually).
    basis:   np.ndarray = field(default=None, repr=False)  # (n_symbols, n_dims)
    centers: np.ndarray = field(default=None, repr=False)  # (n_symbols, d)
    x_mean:  np.ndarray = field(default=None, repr=False)  # (n_dims,)
    x_scale: float      = field(default=1.0,  repr=False)

    def __post_init__(self) -> None:
        if self.basis is None:
            # Default: random orthonormal projection.
            raw = self.rng.standard_normal((self.n_symbols, self.n_dims))
            self.basis = self._orthonormalize(raw)
        if self.centers is None:
            # Default quantization: uniform bins in [-1, 1].
            self._set_uniform_centers(lo=-1.0, hi=1.0)
        if self.x_mean is None:
            self.x_mean = np.zeros(self.n_dims)

    # ---- internal helpers ----

    @staticmethod
    def _orthonormalize(M: np.ndarray) -> np.ndarray:
        """Return row-orthonormal matrix via QR (min(rows, cols) rows)."""
        Q, _ = np.linalg.qr(M.T)
        return Q.T[:M.shape[0]]

    def _set_uniform_centers(self, lo: float, hi: float) -> None:
        """Uniform quantization: d bins per symbol axis."""
        edges = np.linspace(lo, hi, self.d + 1)
        self.centers = np.array(
            [(edges[i] + edges[i + 1]) / 2 for i in range(self.d)]
        )  # shape (d,) -- same bins for all symbols

    # ---- fit (optional -- learns better projection from data) ----

    def fit(self, data: np.ndarray, n_iter: int = 5) -> "QuditCodebook":
        """Fit basis and quantization centers to a data matrix (n_samples, n_dims).

        Uses PCA-style projection (top k principal components) so the
        k retained axes capture maximum variance -- best reconstruction.
        """
        X = np.asarray(data, dtype=np.float64)
        self.x_mean  = X.mean(axis=0)
        Xc           = X - self.x_mean
        self.x_scale = float(np.std(Xc)) or 1.0
        Xn           = Xc / self.x_scale
        # SVD: top k right-singular vectors = principal components.
        _, _, Vt = np.linalg.svd(Xn, full_matrices=False)
        k = min(self.n_symbols, Vt.shape[0])
        self.basis = Vt[:k]
        if k < self.n_symbols:
            # Pad with random orthogonal complement if needed.
            extra = self.rng.standard_normal((self.n_symbols - k, self.n_dims))
            self.basis = np.vstack([self.basis, extra])
            self.basis = self._orthonormalize(self.basis)
        # Fit quantization bins from projected data range.
        proj = Xn @ self.basis.T   # (n_samples, n_symbols)
        lo   = float(proj.min())
        hi   = float(proj.max())
        self._set_uniform_centers(lo=lo, hi=hi)
        return self

    # ---- encode / decode ----

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Map activation vector x (N,) -> qudit symbol array (k,) of ints.

        Each integer is in {0, ..., d-1}.
        """
        x  = np.asarray(x, dtype=np.float64)
        xn = (x - self.x_mean) / self.x_scale
        proj = self.basis @ xn          # (k,)
        # Quantize: find nearest center for each symbol axis.
        # centers shape (d,); we compare each proj[i] against all centers.
        diffs = np.abs(proj[:, None] - self.centers[None, :])  # (k, d)
        return diffs.argmin(axis=1).astype(np.int32)            # (k,)

    def decode(self, symbols: np.ndarray) -> np.ndarray:
        """Map qudit symbol array (k,) -> reconstructed activation (N,).

        Reconstruction = inverse-project the dequantized values.
        """
        symbols = np.asarray(symbols, dtype=np.int32)
        proj    = self.centers[symbols]               # (k,) float values
        x_norm  = self.basis.T @ proj                 # (N,)
        return x_norm * self.x_scale + self.x_mean

    def compression_ratio(self) -> float:
        """Bits of raw float vector / bits in qudit code."""
        raw_bits  = self.n_dims * 32          # 32-bit floats
        code_bits = self.n_symbols * np.log2(self.d)
        return raw_bits / code_bits

    def reconstruction_error(self, x: np.ndarray) -> float:
        """Round-trip RMSE: ||x - decode(encode(x))|| / ||x||."""
        xhat = self.decode(self.encode(x))
        denom = float(np.linalg.norm(x)) or 1.0
        return float(np.linalg.norm(x - xhat)) / denom


# ---------------------------------------------------------------------------
# QuditChannel  (latency + optional noise)
# ---------------------------------------------------------------------------

@dataclass
class QuditChannel:
    """Simulates a qudit transmission channel.

    Parameters
    ----------
    d            : qudit dimension
    symbol_delay : time cost per symbol (ms)  -- models synaptic latency
    spike_delay  : time cost per raw bit (ms) -- baseline for comparison
    error_rate   : probability of a symbol flip (0 = perfect channel)
    rng          : random generator
    """
    d:            int   = 10
    symbol_delay: float = 1.0    # ms per qudit symbol
    spike_delay:  float = 1.0    # ms per raw spike / bit
    error_rate:   float = 0.0
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    def transmit(self, symbols: np.ndarray) -> tuple[np.ndarray, float]:
        """Transmit k qudit symbols.  Returns (received_symbols, latency_ms)."""
        k        = len(symbols)
        latency  = k * self.symbol_delay
        received = symbols.copy()
        if self.error_rate > 0:
            mask = self.rng.random(k) < self.error_rate
            noise = self.rng.integers(0, self.d, size=k)
            received[mask] = noise[mask]
        return received, latency

    def raw_latency(self, n_dims: int) -> float:
        """Latency to send n_dims raw spikes / bits."""
        return n_dims * self.spike_delay

    def speedup(self, n_dims: int, n_symbols: int) -> float:
        """Ratio: raw_latency / relay_latency."""
        return self.raw_latency(n_dims) / max(n_symbols * self.symbol_delay, 1e-9)


# ---------------------------------------------------------------------------
# CorticalRelay  (full pipeline)
# ---------------------------------------------------------------------------

@dataclass
class CorticalRelay:
    """Full encode -> channel -> decode pipeline.

    Usage
    -----
        relay = CorticalRelay(n_dims=64, n_symbols=6, d=10)
        relay.fit(training_data)      # optional: learn better codebook

        received, stats = relay.transmit(activation_vector)
        # stats: latency, compression_ratio, reconstruction_error, speedup
    """
    n_dims:    int
    n_symbols: int
    d:         int   = 10
    symbol_delay: float = 1.0
    spike_delay:  float = 1.0
    error_rate:   float = 0.0
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    codebook: QuditCodebook = field(default=None, repr=False)
    channel:  QuditChannel  = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.codebook is None:
            self.codebook = QuditCodebook(
                n_dims=self.n_dims, n_symbols=self.n_symbols,
                d=self.d, rng=self.rng)
        if self.channel is None:
            self.channel = QuditChannel(
                d=self.d,
                symbol_delay=self.symbol_delay,
                spike_delay=self.spike_delay,
                error_rate=self.error_rate,
                rng=self.rng)

    def fit(self, data: np.ndarray) -> "CorticalRelay":
        """Learn codebook from data matrix (n_samples, n_dims)."""
        self.codebook.fit(data)
        return self

    def transmit(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        """Encode x, transmit via channel, decode, return result + stats.

        Returns
        -------
        x_hat : reconstructed activation vector (N,)
        stats : dict with latency_ms, speedup, compression_ratio,
                reconstruction_error, symbols_sent, raw_bits_equivalent
        """
        x = np.asarray(x, dtype=np.float64)
        # Encode.
        symbols = self.codebook.encode(x)
        # Transmit.
        received, latency = self.channel.transmit(symbols)
        # Decode.
        x_hat = self.codebook.decode(received)
        # Stats.
        stats = {
            "symbols_sent":        int(self.n_symbols),
            "raw_bits_equivalent": int(self.n_dims * 32),
            "latency_ms":          latency,
            "raw_latency_ms":      self.channel.raw_latency(self.n_dims),
            "speedup":             self.channel.speedup(self.n_dims, self.n_symbols),
            "compression_ratio":   self.codebook.compression_ratio(),
            "reconstruction_error": self.codebook.reconstruction_error(x),
            "symbols":             symbols.copy(),
            "received":            received.copy(),
        }
        return x_hat, stats

    def encode_only(self, x: np.ndarray) -> np.ndarray:
        """Return qudit symbol array without transmitting."""
        return self.codebook.encode(x)

    def decode_only(self, symbols: np.ndarray) -> np.ndarray:
        """Reconstruct activation from symbols without a channel."""
        return self.codebook.decode(symbols)


# ---------------------------------------------------------------------------
# RelayProfiler  (benchmark a relay on a stream of activations)
# ---------------------------------------------------------------------------

class RelayProfiler:
    """Run a CorticalRelay on many activations and collect aggregate stats.

    Example
    -------
        profiler = RelayProfiler(relay)
        for x in stream:
            profiler.step(x)
        report = profiler.report()
    """

    def __init__(self, relay: CorticalRelay) -> None:
        self.relay  = relay
        self._errs: list[float] = []
        self._lats: list[float] = []
        self._speedups: list[float] = []
        self._n = 0

    def step(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        x_hat, stats = self.relay.transmit(x)
        self._errs.append(stats["reconstruction_error"])
        self._lats.append(stats["latency_ms"])
        self._speedups.append(stats["speedup"])
        self._n += 1
        return x_hat, stats

    def report(self) -> dict:
        if self._n == 0:
            return {}
        return {
            "n_transmissions":       self._n,
            "mean_reconstruction_error": float(np.mean(self._errs)),
            "std_reconstruction_error":  float(np.std(self._errs)),
            "mean_latency_ms":       float(np.mean(self._lats)),
            "mean_speedup":          float(np.mean(self._speedups)),
            "compression_ratio":     self.relay.codebook.compression_ratio(),
            "n_dims":                self.relay.n_dims,
            "n_symbols":             self.relay.n_symbols,
            "d":                     self.relay.d,
        }

    def reset(self) -> None:
        self._errs.clear()
        self._lats.clear()
        self._speedups.clear()
        self._n = 0


# ---------------------------------------------------------------------------
# Convenience: connect two brain-module callables through the relay
# ---------------------------------------------------------------------------

def relay_connect(
    region_a: Callable[[np.ndarray], np.ndarray],
    region_b: Callable[[np.ndarray], np.ndarray],
    relay: CorticalRelay,
) -> Callable[[np.ndarray], tuple[np.ndarray, dict]]:
    """Wire region_a -> relay -> region_b into a single callable.

    Returns a function f(x) that:
      1. Runs region_a(x) to get activation
      2. Compresses and transmits via relay
      3. Feeds reconstructed signal into region_b
      4. Returns (region_b output, relay stats)
    """
    def pipeline(x: np.ndarray) -> tuple[np.ndarray, dict]:
        act_a        = region_a(x)
        act_b_input, stats = relay.transmit(act_a)
        out          = region_b(act_b_input)
        return out, stats
    return pipeline
