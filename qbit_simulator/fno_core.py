"""Fourier Neural Operator (FNO) -- core architecture.

Based on: Li et al. "Fourier Neural Operator for Parametric PDEs" (2021)
arXiv: https://arxiv.org/abs/2010.08895

How a single FNO layer works
-----------------------------
  1. LIFT      x (n, d_in)  ->  h (n, d_model)   [pointwise linear]
  2. SPECTRAL  h  ->  FFT  ->  learned weights on k_max modes  ->  iFFT
  3. RESIDUAL  h  ->  pointwise linear  (bypass around spectral)
  4. SUM       spectral_out + residual_out
  5. ACTIVATE  GeLU / ReLU
  Repeat L times, then PROJECT h (n, d_model) -> y (n, d_out)

Key properties
--------------
  Resolution invariance : trained on n points, evaluates on any n' points
  Mesh independence     : works on irregular or variable-length grids
  Speed                 : once trained, inference is O(n log n) via FFT

Numpy-only forward pass
-----------------------
This file implements the full FNO forward pass using only numpy + scipy.
Weights are stored as numpy arrays and can be:
  - randomly initialised (useful for testing shapes / wiring)
  - loaded from a .npz checkpoint saved after PyTorch training
  - replaced with your own trained arrays

Training interface
------------------
`FNO1d.fit()` requires PyTorch (optional dependency).  Import will
gracefully fail and raise InstructionError with a helpful message if
torch is not installed.  Inference (`forward()`) always works with numpy.

Classes
-------
  SpectralConv1d   -- single spectral convolution layer
  FNOBlock         -- spectral conv + residual + activation
  FNO1d            -- full FNO: lift -> L blocks -> project
  FNOCheckpoint    -- save / load weight snapshots
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import os


# ---------------------------------------------------------------------------
# SpectralConv1d
# ---------------------------------------------------------------------------

class SpectralConv1d:
    """Single spectral convolution over the first k_max Fourier modes.

    Parameters
    ----------
    d_in    : input channel width
    d_out   : output channel width
    k_max   : number of Fourier modes to keep (low-frequency filter)
    seed    : weight initialisation seed
    """

    def __init__(self, d_in: int, d_out: int, k_max: int,
                 seed: int = 0) -> None:
        self.d_in  = d_in
        self.d_out = d_out
        self.k_max = k_max

        rng   = np.random.default_rng(seed)
        scale = 1.0 / (d_in * d_out)
        # Complex weights: (k_max, d_in, d_out)
        self.W_re = rng.uniform(-scale, scale, (k_max, d_in, d_out))
        self.W_im = rng.uniform(-scale, scale, (k_max, d_in, d_out))

    @property
    def W(self) -> np.ndarray:
        """Complex weight tensor (k_max, d_in, d_out)."""
        return self.W_re + 1j * self.W_im

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Spectral convolution.

        Parameters
        ----------
        x : (n_points, d_in)  -- signal in spatial domain

        Returns
        -------
        out : (n_points, d_out)  -- filtered signal in spatial domain
        """
        n, d_in = x.shape
        assert d_in == self.d_in, f"expected d_in={self.d_in}, got {d_in}"

        # FFT along spatial axis.
        x_ft = np.fft.rfft(x, axis=0)           # (n//2+1, d_in)

        k = min(self.k_max, x_ft.shape[0])
        out_ft = np.zeros((x_ft.shape[0], self.d_out), dtype=complex)

        # Learned linear transform on first k modes: Einstein sum over d_in.
        # out_ft[m, j] = sum_i  x_ft[m, i] * W[m, i, j]
        out_ft[:k] = np.einsum("mi,mij->mj", x_ft[:k], self.W[:k])

        # Inverse FFT back to spatial domain.
        out = np.fft.irfft(out_ft, n=n, axis=0)  # (n, d_out)
        return out

    def param_count(self) -> int:
        return 2 * self.k_max * self.d_in * self.d_out

    def state_dict(self) -> dict:
        return {"W_re": self.W_re, "W_im": self.W_im,
                "d_in": self.d_in, "d_out": self.d_out, "k_max": self.k_max}

    def load_state_dict(self, sd: dict) -> None:
        self.W_re  = sd["W_re"]
        self.W_im  = sd["W_im"]
        self.d_in  = int(sd["d_in"])
        self.d_out = int(sd["d_out"])
        self.k_max = int(sd["k_max"])


# ---------------------------------------------------------------------------
# FNOBlock
# ---------------------------------------------------------------------------

class FNOBlock:
    """One FNO residual block: spectral conv + pointwise residual + activation.

    out = activation( SpectralConv(x) + W_res @ x )

    Parameters
    ----------
    d_model    : channel width (same in and out)
    k_max      : Fourier modes to keep
    activation : 'gelu' | 'relu' | 'tanh'
    seed       : weight init seed
    """

    def __init__(self, d_model: int, k_max: int,
                 activation: str = "gelu", seed: int = 0) -> None:
        self.d_model    = d_model
        self.k_max      = k_max
        self.activation = activation

        rng          = np.random.default_rng(seed)
        self.spec    = SpectralConv1d(d_model, d_model, k_max, seed=seed)
        # Pointwise residual (no bias for simplicity).
        scale        = np.sqrt(2.0 / d_model)
        self.W_res   = rng.standard_normal((d_model, d_model)) * scale

    def _act(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "gelu":
            # Approximate GeLU.
            return x * 0.5 * (1.0 + np.tanh(
                np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))
        if self.activation == "relu":
            return np.maximum(x, 0.0)
        if self.activation == "tanh":
            return np.tanh(x)
        raise ValueError(f"unknown activation: {self.activation}")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x : (n_points, d_model) -> (n_points, d_model)"""
        spectral  = self.spec.forward(x)          # (n, d_model)
        residual  = x @ self.W_res.T              # (n, d_model)
        return self._act(spectral + residual)

    def param_count(self) -> int:
        return self.spec.param_count() + self.d_model ** 2

    def state_dict(self) -> dict:
        return {"spec": self.spec.state_dict(), "W_res": self.W_res,
                "d_model": self.d_model, "k_max": self.k_max,
                "activation": self.activation}

    def load_state_dict(self, sd: dict) -> None:
        self.spec.load_state_dict(sd["spec"])
        self.W_res     = sd["W_res"]
        self.d_model   = int(sd["d_model"])
        self.k_max     = int(sd["k_max"])
        self.activation = str(sd["activation"])


# ---------------------------------------------------------------------------
# FNO1d
# ---------------------------------------------------------------------------

class FNO1d:
    """Full 1-D Fourier Neural Operator.

    Architecture:
        x  (n, d_in)
        -> Lifting    W_lift  (d_in  -> d_model)
        -> FNOBlock x n_layers
        -> Projection W_proj  (d_model -> d_out)
        -> y  (n, d_out)

    Parameters
    ----------
    d_in      : input channel width (e.g. signal dimensionality)
    d_out     : output channel width
    d_model   : internal width (hidden channels)
    n_layers  : number of FNO blocks
    k_max     : Fourier modes retained per layer
    activation: nonlinearity in each block
    seed      : reproducibility
    """

    def __init__(self, d_in: int, d_out: int, d_model: int = 64,
                 n_layers: int = 4, k_max: int = 16,
                 activation: str = "gelu", seed: int = 0) -> None:
        self.d_in     = d_in
        self.d_out    = d_out
        self.d_model  = d_model
        self.n_layers = n_layers
        self.k_max    = k_max

        rng = np.random.default_rng(seed)
        s   = np.sqrt(2.0 / d_model)

        self.W_lift = rng.standard_normal((d_in,    d_model)) * s
        self.W_proj = rng.standard_normal((d_model, d_out))   * s
        self.b_proj = np.zeros(d_out)

        self.blocks = [
            FNOBlock(d_model, k_max, activation=activation, seed=seed + i)
            for i in range(n_layers)
        ]

    # ------------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Full forward pass.

        Parameters
        ----------
        x : (n_points, d_in) or (d_in,) for a single point

        Returns
        -------
        y : (n_points, d_out)
        """
        x = np.asarray(x, dtype=np.float64)
        scalar = x.ndim == 1
        if scalar:
            x = x[np.newaxis, :]   # (1, d_in)

        h = x @ self.W_lift        # (n, d_model)
        for block in self.blocks:
            h = block.forward(h)
        y = h @ self.W_proj + self.b_proj   # (n, d_out)

        return y[0] if scalar else y

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    # ------------------------------------------------------------------
    def param_count(self) -> int:
        n = self.d_in * self.d_model + self.d_model * self.d_out + self.d_out
        return n + sum(b.param_count() for b in self.blocks)

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "W_lift":   self.W_lift,
            "W_proj":   self.W_proj,
            "b_proj":   self.b_proj,
            "blocks":   [b.state_dict() for b in self.blocks],
            "config":   {
                "d_in": self.d_in, "d_out": self.d_out,
                "d_model": self.d_model, "n_layers": self.n_layers,
                "k_max": self.k_max,
            },
        }

    def load_state_dict(self, sd: dict) -> None:
        self.W_lift = sd["W_lift"]
        self.W_proj = sd["W_proj"]
        self.b_proj = sd["b_proj"]
        for block, bsd in zip(self.blocks, sd["blocks"]):
            block.load_state_dict(bsd)

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, Y: np.ndarray,
            n_epochs: int = 100, lr: float = 1e-3,
            batch_size: int = 32) -> list[float]:
        """Train via PyTorch (optional dependency).

        Parameters
        ----------
        X : (N, n_points, d_in)  input trajectories
        Y : (N, n_points, d_out) target outputs

        Returns
        -------
        losses : list of per-epoch MSE loss values
        """
        try:
            import torch
            return self._fit_torch(X, Y, n_epochs, lr, batch_size)
        except ImportError:
            raise ImportError(
                "PyTorch is required for FNO training.  Install it with:\n"
                "  pip install torch\n"
                "Inference (forward()) works without PyTorch.")

    def _fit_torch(self, X, Y, n_epochs, lr, batch_size) -> list[float]:
        import torch
        import torch.nn as nn
        import torch.optim as optim

        # Wrap numpy weights into a tiny torch model mirroring this FNO.
        # For a full-featured training loop use the neuraloperator library:
        # https://github.com/neuraloperator/neuraloperator
        raise NotImplementedError(
            "Full PyTorch training loop not yet implemented here.\n"
            "Recommended: use the neuraloperator library, train there,\n"
            "then load weights back with FNO1d.load_from_npz().")

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save weights to a .npz file."""
        sd   = self.state_dict()
        flat = {}
        flat["W_lift"] = sd["W_lift"]
        flat["W_proj"] = sd["W_proj"]
        flat["b_proj"] = sd["b_proj"]
        for i, bsd in enumerate(sd["blocks"]):
            flat[f"block{i}_W_res"]     = bsd["W_res"]
            flat[f"block{i}_spec_W_re"] = bsd["spec"]["W_re"]
            flat[f"block{i}_spec_W_im"] = bsd["spec"]["W_im"]
        # Save config as 1-D arrays (npz only stores arrays).
        cfg = sd["config"]
        for k, v in cfg.items():
            flat[f"cfg_{k}"] = np.array([v])
        np.savez(path, **flat)

    @classmethod
    def load(cls, path: str) -> "FNO1d":
        """Load a previously saved FNO from a .npz file."""
        data = np.load(path, allow_pickle=False)
        cfg  = {k[4:]: int(data[k][0])
                for k in data.files if k.startswith("cfg_")}
        fno  = cls(**cfg)
        fno.W_lift = data["W_lift"]
        fno.W_proj = data["W_proj"]
        fno.b_proj = data["b_proj"]
        for i, block in enumerate(fno.blocks):
            block.W_res                  = data[f"block{i}_W_res"]
            block.spec.W_re              = data[f"block{i}_spec_W_re"]
            block.spec.W_im              = data[f"block{i}_spec_W_im"]
        return fno

    def __repr__(self) -> str:
        return (f"FNO1d(d_in={self.d_in}, d_out={self.d_out}, "
                f"d_model={self.d_model}, layers={self.n_layers}, "
                f"k_max={self.k_max}, params={self.param_count():,})")
