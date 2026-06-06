"""Liquid State Machine / Reservoir computing.

Reservoir computing (Maass-Natschläger-Markram 2002 / Jaeger 2001):
a large RANDOM recurrent neural network ("reservoir") + a simple
TRAINABLE LINEAR READOUT. Despite the randomness, the reservoir's
dynamics project an input stream onto a high-dimensional state space
where complex tasks become linearly separable.

Two flavors:
  - **Echo state network (ESN)**: rate-based, continuous neurons.
  - **Liquid state machine (LSM)**: spiking neurons (LIF).

This module provides:

  - `EchoStateNetwork(n, spectral_radius)`: classical ESN with tanh
    nonlinearity.
  - `LiquidStateMachine(n, sparsity)`: SNN reservoir using LIF neurons.
  - `train_readout(activity_traces, targets)`: ridge regression for the
    linear readout.
  - Demonstration: classify pulse-train inputs (rate-coded XOR with
    temporal structure).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .lif import LIFPopulation


# ----------------------------------------------------------------------------
# Echo State Network (rate-based reservoir)
# ----------------------------------------------------------------------------

@dataclass
class EchoStateNetwork:
    """Classical ESN. Activation: x_t = (1−α) x_{t−1} + α tanh(W_res x_{t−1} + W_in u_t).

    Args:
        n:               reservoir size.
        n_input:         input dimension.
        spectral_radius: scaling of W_res (must be < 1 for echo-state property).
        sparsity:        fraction of nonzero W_res entries.
        alpha:           leak rate.
        rng:             generator.
    """
    n:               int = 100
    n_input:         int = 1
    spectral_radius: float = 0.9
    sparsity:        float = 0.1
    alpha:           float = 0.3
    seed:            int = 0
    W_res:  np.ndarray = field(default=None, repr=False)
    W_in:   np.ndarray = field(default=None, repr=False)
    state:  np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        if self.W_res is None:
            W = rng.normal(size=(self.n, self.n))
            mask = rng.uniform(size=(self.n, self.n)) < self.sparsity
            W = W * mask
            # Rescale to target spectral radius.
            current_sr = max(abs(np.linalg.eigvals(W)))
            if current_sr > 1e-9:
                W = W * (self.spectral_radius / current_sr)
            self.W_res = W
        if self.W_in is None:
            self.W_in = rng.normal(scale=1.0, size=(self.n, self.n_input))
        if self.state is None:
            self.state = np.zeros(self.n)

    def reset(self) -> None:
        self.state[:] = 0

    def step(self, u: np.ndarray) -> np.ndarray:
        """One step. `u` shape (n_input,)."""
        pre = self.W_res @ self.state + self.W_in @ u
        new = np.tanh(pre)
        self.state = (1 - self.alpha) * self.state + self.alpha * new
        return self.state.copy()

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """Run on a sequence of inputs. inputs shape: (T, n_input).
        Returns activity_traces of shape (T, n)."""
        T = inputs.shape[0]
        out = np.zeros((T, self.n))
        for t in range(T):
            out[t] = self.step(inputs[t])
        return out


# ----------------------------------------------------------------------------
# Liquid State Machine (spiking reservoir)
# ----------------------------------------------------------------------------

@dataclass
class LiquidStateMachine:
    """SNN reservoir. Each neuron is LIF; connectivity is random and sparse.

    Output is the smoothed spike train (low-pass filtered for stability).
    """
    n:               int = 80
    n_input:         int = 1
    sparsity:        float = 0.1
    w_scale:         float = 0.8
    syn_tau:         float = 25.0
    seed:            int = 0

    W_res:  np.ndarray = field(default=None, repr=False)
    W_in:   np.ndarray = field(default=None, repr=False)
    pop:    LIFPopulation = field(default=None)
    syn_current: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        if self.W_res is None:
            W = rng.normal(size=(self.n, self.n))
            mask = rng.uniform(size=(self.n, self.n)) < self.sparsity
            W = W * mask * self.w_scale
            np.fill_diagonal(W, 0)
            self.W_res = W
        if self.W_in is None:
            self.W_in = rng.normal(scale=1.0, size=(self.n, self.n_input))
        if self.pop is None:
            self.pop = LIFPopulation(n=self.n)
        if self.syn_current is None:
            self.syn_current = np.zeros(self.n)

    def reset(self) -> None:
        self.pop.reset()
        self.syn_current[:] = 0

    def step(self, u: np.ndarray, t: int = 0) -> np.ndarray:
        """One step. Returns the binary spike vector."""
        self.syn_current *= np.exp(-1.0 / self.syn_tau)
        ext = self.W_in @ u + self.syn_current
        spikes = self.pop.step(ext, t=t)
        # Re-circulate via W_res.
        self.syn_current += self.W_res @ spikes.astype(float)
        return spikes

    def run(self, inputs: np.ndarray, smoothing_tau: float = 10.0
              ) -> np.ndarray:
        """Run on inputs and return SMOOTHED firing-rate traces.

        inputs shape (T, n_input). Output shape (T, n).
        """
        T = inputs.shape[0]
        spikes_history = np.zeros((T, self.n))
        smooth = np.zeros(self.n)
        decay = np.exp(-1.0 / smoothing_tau)
        for t in range(T):
            s = self.step(inputs[t], t=t)
            smooth = smooth * decay + s.astype(float)
            spikes_history[t] = smooth
        return spikes_history


# ----------------------------------------------------------------------------
# Readout layer: ridge regression
# ----------------------------------------------------------------------------

def train_readout(
    activity_traces: np.ndarray, targets: np.ndarray,
    ridge: float = 1e-3,
) -> np.ndarray:
    """Solve (X^T X + λI) W = X^T y for the linear readout weights.

    Args:
        activity_traces: shape (T, n_features).
        targets:         shape (T,) or (T, n_outputs).
        ridge:           Tikhonov regularization.

    Returns:
        W: shape (n_features,) or (n_features, n_outputs).
    """
    X = activity_traces
    y = targets
    n_features = X.shape[1]
    XtX = X.T @ X + ridge * np.eye(n_features)
    Xty = X.T @ y
    return np.linalg.solve(XtX, Xty)


def predict(activity_traces: np.ndarray, W: np.ndarray) -> np.ndarray:
    return activity_traces @ W
