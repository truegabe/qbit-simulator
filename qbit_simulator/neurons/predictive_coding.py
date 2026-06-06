"""Predictive coding: hierarchical Bayesian inference via prediction-error
minimization.

Karl Friston's free-energy principle (2005-ongoing) proposes that the
brain operates as a hierarchical Bayesian inference engine. At each
level, the brain maintains:

  - A "belief" / latent state x_l about what's happening at level l.
  - A "prediction" of the next level's state: x_{l-1} ≈ f(x_l).
  - A "prediction error" e_{l-1} = (observed) - (predicted) at that level.

Predictive coding makes this concrete with two neuron populations per
level:

  - **State units** x_l: estimate the latent variable.
  - **Error units** e_l: report (input − prediction).

Update equations (Rao & Ballard 1999):

    dx_l/dt    =  W_{l+1}^T · e_{l+1}  −  e_l
    e_l        =  x_l  −  f(W_l · x_{l+1})

The lowest-level error is driven by the SENSORY INPUT minus the top-down
prediction. Iterating these equations performs gradient descent on
prediction error (variational free energy).

Learning: weights W_l are updated to minimize prediction error:
    ΔW_l ∝ e_{l-1} · x_l^T

This module provides:

  - `PredictiveCodingLayer`: a single level with state + error units.
  - `PredictiveCodingNetwork(layer_sizes)`: a hierarchical PC network.
  - `.infer(sensory_input, n_iter)`: run the inference dynamics.
  - `.learn(sensory_input, n_iter, lr)`: weight updates after inference.
  - `.predict_top_down(top_state)`: generate from a top-level state
    (the "hallucination" mode).

We use a TOY linear-Gaussian PC formulation: f is the identity / linear
mapping, so predictions are W · x. This is sufficient to demonstrate the
core dynamics; non-linear extensions (sigmoid, softplus) plug in directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ----------------------------------------------------------------------------
# Single layer
# ----------------------------------------------------------------------------

@dataclass
class PredictiveCodingLayer:
    """One level of a PC hierarchy.

    Attributes:
        n_state:  number of state units at THIS level.
        n_below:  number of state units at the LOWER level (what we
                  predict downward).
        W:        weight matrix shape (n_below, n_state) — generative
                  weights mapping our state → prediction of below.
        x:        current state (n_state,).
        e:        current prediction error (n_below,).
    """
    n_state: int
    n_below: int
    W:       np.ndarray = field(default=None, repr=False)
    x:       np.ndarray = field(default=None, repr=False)
    e:       np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.W is None:
            self.W = np.random.default_rng(0).normal(
                scale=0.1, size=(self.n_below, self.n_state)
            )
        if self.x is None:
            self.x = np.zeros(self.n_state)
        if self.e is None:
            self.e = np.zeros(self.n_below)

    def predict_below(self) -> np.ndarray:
        """Top-down prediction for the level below: x_below_hat = W · x."""
        return self.W @ self.x


# ----------------------------------------------------------------------------
# Hierarchical PC network
# ----------------------------------------------------------------------------

@dataclass
class PredictiveCodingNetwork:
    """A hierarchical predictive-coding network.

    layer_sizes = [n_input, n_hidden_1, n_hidden_2, ..., n_top].

    Internally we build len(layer_sizes) - 1 layers:
      - layer[l] has n_state = layer_sizes[l+1] state units and
        predicts down to n_below = layer_sizes[l] units.

    sensory_input arrives at the BOTTOM (level 0).
    """
    layer_sizes: list[int]
    layers: list[PredictiveCodingLayer] = field(default_factory=list)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(0)
        if not self.layers:
            for l in range(len(self.layer_sizes) - 1):
                self.layers.append(PredictiveCodingLayer(
                    n_state=self.layer_sizes[l + 1],
                    n_below=self.layer_sizes[l],
                    W=rng.normal(scale=0.1, size=(self.layer_sizes[l],
                                                    self.layer_sizes[l + 1])),
                ))

    def reset_states(self) -> None:
        for layer in self.layers:
            layer.x[:] = 0
            layer.e[:] = 0

    # ---- Inference (state dynamics) ----

    def infer(
        self, sensory_input: np.ndarray, n_iter: int = 50,
        lr_x: float = 0.1,
    ) -> dict:
        """Iterate state-unit dynamics until convergence (gradient
        descent on free energy).

        Args:
            sensory_input: shape (layer_sizes[0],).
            n_iter:        number of inference steps.
            lr_x:          state-update learning rate.

        Returns:
            dict with the final state at each layer + final errors +
            free-energy history.
        """
        if sensory_input.shape != (self.layer_sizes[0],):
            raise ValueError(
                f"sensory_input shape {sensory_input.shape} != "
                f"({self.layer_sizes[0]},)"
            )

        errors_below = [None] * len(self.layers)
        F_history = []
        for it in range(n_iter):
            # Bottom-up: compute prediction errors at each level.
            below = sensory_input
            for l, layer in enumerate(self.layers):
                pred = layer.predict_below()
                e_l = below - pred
                layer.e = e_l
                errors_below[l] = e_l
                below = layer.x
            # Top-down: update each state x_l.
            for l, layer in enumerate(self.layers):
                # ∂F/∂x_l = -W_l^T e_l + e_{l+1}_for_this_layer_as_below
                # where e_{l+1} = x_l - prediction_from_above = -e (if any).
                grad = -layer.W.T @ layer.e
                if l < len(self.layers) - 1:
                    # The level above predicts x_l; its error is
                    # x_l - prediction = -errors_below[l+1].
                    grad = grad + errors_below[l + 1]
                layer.x = layer.x - lr_x * grad
            # Free-energy proxy: total squared prediction error.
            F = 0.5 * sum(np.sum(layer.e ** 2) for layer in self.layers)
            F_history.append(F)
        return {
            "states":          [layer.x.copy() for layer in self.layers],
            "errors":          [layer.e.copy() for layer in self.layers],
            "free_energy":     F_history,
            "n_iter":          n_iter,
        }

    # ---- Learning (weight updates) ----

    def learn(
        self, sensory_input: np.ndarray, n_iter: int = 50,
        lr_x: float = 0.1, lr_w: float = 0.01,
    ) -> dict:
        """Run inference, then apply ONE weight-update step:
            ΔW_l = lr_w · e_{below} · x_l^T

        Returns inference output + a snapshot of the updated weights.
        """
        out = self.infer(sensory_input, n_iter=n_iter, lr_x=lr_x)
        for l, layer in enumerate(self.layers):
            dW = np.outer(layer.e, layer.x)
            layer.W = layer.W + lr_w * dW
        out["weights_after"] = [layer.W.copy() for layer in self.layers]
        return out

    # ---- Top-down generation ----

    def predict_top_down(self, top_state: np.ndarray) -> np.ndarray:
        """Generate a sensory-level prediction from a TOP-level state.

        Useful as a "hallucination" / imagination mode: clamp the highest
        x_l, propagate all predictions down.
        """
        top_layer = self.layers[-1]
        if top_state.shape != (top_layer.n_state,):
            raise ValueError(
                f"top_state shape {top_state.shape} != "
                f"({top_layer.n_state},)"
            )
        x = top_state.copy()
        # Walk down from the top layer to layer 0.
        for layer in reversed(self.layers):
            x = layer.W @ x
        return x


# ----------------------------------------------------------------------------
# Free-energy diagnostic
# ----------------------------------------------------------------------------

def free_energy(network: PredictiveCodingNetwork) -> float:
    """Current free-energy proxy: 0.5 · sum_l ||e_l||²."""
    return 0.5 * float(sum(np.sum(layer.e ** 2) for layer in network.layers))


# ----------------------------------------------------------------------------
# Training driver: many samples
# ----------------------------------------------------------------------------

def train_predictive_coding(
    network: PredictiveCodingNetwork,
    dataset: np.ndarray,
    n_epochs: int = 10,
    n_iter_per_sample: int = 30,
    lr_x: float = 0.1,
    lr_w: float = 0.01,
    rng: np.random.Generator | None = None,
) -> dict:
    """Train a PC network on a batch of sensory samples.

    For each sample:
      1. Reset state units (NOT weights).
      2. Run inference to settle states.
      3. Update weights to reduce residual error.

    Returns:
        dict with epoch-wise mean free energy.
    """
    rng = rng or np.random.default_rng()
    history = []
    n_samples = dataset.shape[0]
    for epoch in range(n_epochs):
        order = rng.permutation(n_samples)
        epoch_F = []
        for idx in order:
            network.reset_states()
            out = network.learn(
                dataset[idx], n_iter=n_iter_per_sample,
                lr_x=lr_x, lr_w=lr_w,
            )
            epoch_F.append(out["free_energy"][-1])
        history.append(float(np.mean(epoch_F)))
    return {"mean_free_energy_per_epoch": history}
