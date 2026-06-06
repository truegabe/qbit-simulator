"""Predictive relay -- transmit only the surprise.

The brain does not send full activation vectors between regions.
It sends PREDICTION ERRORS -- the difference between what was expected
and what actually arrived.  Predictions flow DOWN the hierarchy;
errors flow UP.  (Rao & Ballard 1999; Friston 2010 Free Energy.)

Why this matters for speed:
  - On a stable input (you're staring at the same wall) the error is ~0.
    Almost nothing travels over the channel.
  - On a novel input (something moves) the full error burst fires.
  - Average channel traffic scales with INPUT VARIABILITY, not input size.

Architecture:
  Sender side:
    1. Receive new signal x_t
    2. Compute error e_t = x_t - x_hat_t   (x_hat = current prediction)
    3. Transmit only e_t  (often sparse / near-zero)
    4. Update prediction: x_hat_{t+1} = predict(x_t)

  Receiver side:
    1. Receive error e_t
    2. Reconstruct: x_hat_t + e_t  = x_t  (exact recovery)
    3. Update own prediction: x_hat_{t+1} = predict(x_hat_t + e_t)

Both sides keep synchronized predictions -- they diverge only when the
channel is noisy or the predictor lags behind a fast-changing input.

Predictors provided:
  EMAPredictor       -- exponential moving average (zero parameters)
  LinearPredictor    -- learned W (online gradient descent)
  ConstantPredictor  -- always predicts the mean (baseline)

Classes
-------
  EMAPredictor
  LinearPredictor
  ConstantPredictor
  PredictiveRelay    -- sender + receiver with matching predictors
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


# ---------------------------------------------------------------------------
# Predictor protocol
# ---------------------------------------------------------------------------

class Predictor(Protocol):
    def predict(self, x: np.ndarray) -> np.ndarray: ...
    def update(self, x: np.ndarray) -> None: ...
    def reset(self) -> None: ...


# ---------------------------------------------------------------------------
# Concrete predictors
# ---------------------------------------------------------------------------

@dataclass
class EMAPredictor:
    """Exponential moving average predictor.

    x_hat_{t+1} = alpha * x_t + (1 - alpha) * x_hat_t
    """
    n_dims: int
    alpha:  float = 0.2     # smoothing factor (0=static, 1=no memory)
    _x_hat: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._x_hat = np.zeros(self.n_dims)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._x_hat.copy()

    def update(self, x: np.ndarray) -> None:
        self._x_hat = self.alpha * x + (1 - self.alpha) * self._x_hat

    def reset(self) -> None:
        self._x_hat[:] = 0.0


@dataclass
class LinearPredictor:
    """Learned linear predictor: x_hat = W @ x_prev + b

    Updated online with gradient descent on MSE.
    """
    n_dims: int
    lr:     float = 0.01
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)
    W: np.ndarray = field(default=None, repr=False)
    b: np.ndarray = field(default=None, repr=False)
    _x_prev: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        scale = 0.1 / np.sqrt(self.n_dims)
        self.W = self.rng.standard_normal((self.n_dims, self.n_dims)) * scale
        self.b = np.zeros(self.n_dims)
        self._x_prev = np.zeros(self.n_dims)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.W @ self._x_prev + self.b

    def update(self, x: np.ndarray) -> None:
        x_hat = self.predict(self._x_prev)
        err   = x - x_hat
        # Gradient step.
        self.W += self.lr * np.outer(err, self._x_prev)
        self.b += self.lr * err
        self._x_prev = x.copy()

    def reset(self) -> None:
        self._x_prev[:] = 0.0


@dataclass
class ConstantPredictor:
    """Always predicts the running mean (baseline)."""
    n_dims: int
    _mean:  np.ndarray = field(default=None, repr=False)
    _n:     int        = field(default=0,    repr=False)

    def __post_init__(self) -> None:
        self._mean = np.zeros(self.n_dims)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._mean.copy()

    def update(self, x: np.ndarray) -> None:
        self._n  += 1
        self._mean += (x - self._mean) / self._n

    def reset(self) -> None:
        self._mean[:] = 0.0
        self._n = 0


# ---------------------------------------------------------------------------
# PredictiveRelay
# ---------------------------------------------------------------------------

@dataclass
class PredictiveRelay:
    """Transmit only prediction errors between two brain regions.

    Both sender and receiver must use the SAME predictor type and
    parameters so their predictions stay synchronized.

    Parameters
    ----------
    n_dims          : signal dimensionality
    predictor_type  : 'ema' | 'linear' | 'constant'
    alpha           : EMA smoothing (only for ema)
    lr              : learning rate (only for linear)
    sparsify        : if True, zero-out error components below threshold
    sparse_threshold: fraction of max |error| below which to zero
    """
    n_dims:           int
    predictor_type:   str   = "ema"
    alpha:            float = 0.2
    lr:               float = 0.01
    sparsify:         bool  = False
    sparse_threshold: float = 0.1
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    _sender_pred:   object = field(default=None, repr=False)
    _receiver_pred: object = field(default=None, repr=False)
    _n_sent:        int    = field(default=0, repr=False)
    _total_error_norm: float = field(default=0.0, repr=False)
    _total_signal_norm: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        self._sender_pred   = self._make_predictor()
        self._receiver_pred = self._make_predictor()

    def _make_predictor(self):
        if self.predictor_type == "ema":
            return EMAPredictor(self.n_dims, alpha=self.alpha)
        if self.predictor_type == "linear":
            return LinearPredictor(self.n_dims, lr=self.lr, rng=self.rng)
        if self.predictor_type == "constant":
            return ConstantPredictor(self.n_dims)
        raise ValueError(f"unknown predictor: {self.predictor_type}")

    # ---- sender side ----

    def send(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        """Compute and return the error signal to transmit.

        Returns
        -------
        error  : np.ndarray -- what gets sent over the channel
        stats  : dict       -- compression metrics
        """
        x     = np.asarray(x, dtype=np.float64)
        x_hat = self._sender_pred.predict(x)
        error = x - x_hat

        if self.sparsify:
            thresh = self.sparse_threshold * float(np.abs(error).max() + 1e-12)
            error  = np.where(np.abs(error) >= thresh, error, 0.0)

        self._sender_pred.update(x)
        self._n_sent += 1
        e_norm = float(np.linalg.norm(error))
        x_norm = float(np.linalg.norm(x)) or 1.0
        self._total_error_norm  += e_norm
        self._total_signal_norm += x_norm

        nnz = int(np.count_nonzero(error))
        stats = {
            "error_norm":         e_norm,
            "signal_norm":        x_norm,
            "relative_error":     e_norm / x_norm,
            "sparsity":           1.0 - nnz / self.n_dims,
            "nonzero_components": nnz,
            "prediction":         x_hat,
        }
        return error, stats

    # ---- receiver side ----

    def receive(self, error: np.ndarray) -> np.ndarray:
        """Reconstruct the original signal from the received error.

        Returns the reconstructed activation vector.
        """
        error = np.asarray(error, dtype=np.float64)
        x_hat = self._receiver_pred.predict(error)   # receiver's prediction
        x_rec = x_hat + error
        self._receiver_pred.update(x_rec)
        return x_rec

    # ---- combined (same process, for testing) ----

    def relay(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        """Send then immediately receive (single-process simulation)."""
        error, stats = self.send(x)
        x_rec        = self.receive(error)
        stats["reconstruction_error"] = float(
            np.linalg.norm(x - x_rec) / (np.linalg.norm(x) + 1e-12))
        return x_rec, stats

    # ---- diagnostics ----

    def mean_relative_error(self) -> float:
        """Mean ||error|| / ||signal|| across all transmissions."""
        if self._n_sent == 0:
            return 0.0
        return self._total_error_norm / max(self._total_signal_norm, 1e-12)

    def reset_stats(self) -> None:
        self._n_sent = 0
        self._total_error_norm  = 0.0
        self._total_signal_norm = 0.0
