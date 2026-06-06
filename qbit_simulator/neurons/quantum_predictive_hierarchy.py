"""Quantum predictive-coding hierarchy.

Stacked predictive-coding networks (Rao & Ballard, Friston) infer
latent causes layer-by-layer:

    r_0 = x  (sensory)
    r_1 = inference(r_0; W_0)
    r_2 = inference(r_1; W_1)
    ...

Each layer's inference is a variational optimization. This module
**replaces the classical inference at EACH layer** with a quantum
variational posterior: a parameterized quantum state |ψ_l(θ_l)⟩ that
encodes the layer's belief about r_l.

The hierarchical structure means:
  - Sensory level is concrete (classical, comes from `retina` / `v1`).
  - Each higher level is a quantum belief, fit by minimizing local
    quantum free energy ⟨ψ_l|H_l|ψ_l⟩.
  - The Hamiltonian H_l for layer l is built from the LOWER layer's
    "best explanation" via the generative weight W_{l-1}: it
    rewards latent values that predict the lower layer's mean state.

This gives a true cortical-hierarchy analog where each level's belief
is a quantum distribution — and information flow up/down the
hierarchy uses the quantum free-energy bridge already in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .quantum_variational_pc import (
    QuantumVariationalPC, HardwareEfficientAnsatz, pc_hamiltonian,
)


@dataclass
class QuantumPredictiveHierarchy:
    """Stacked quantum predictive-coding hierarchy.

    layer_sizes:  [n_input, n_h1, n_h2, ...]   (powers of two for qubit counts)
    weights:      [W_0, W_1, ...]   each W_i shape (n_below, n_above)
    """
    layer_sizes: list
    n_layers_quantum: int = None        # number of quantum layers (top first)
    Ws: list = field(default_factory=list)
    qpc_layers: list = field(default_factory=list)
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    def __post_init__(self) -> None:
        L = len(self.layer_sizes)
        if self.n_layers_quantum is None:
            self.n_layers_quantum = L - 1
        if not self.Ws:
            for l in range(L - 1):
                n_below = self.layer_sizes[l]
                n_above = self.layer_sizes[l + 1]
                self.Ws.append(self.rng.normal(0, 1.0 / np.sqrt(n_above),
                                                  (n_below, n_above)))
        # Build one QuantumVariationalPC per inter-layer relation.
        # Each layer's QVPC infers the "best latent in the upper layer"
        # that explains the lower layer's state.
        if not self.qpc_layers:
            for l in range(L - 1):
                n_above = self.layer_sizes[l + 1]
                n_qubits = int(np.ceil(np.log2(max(2, n_above))))
                self.qpc_layers.append(
                    QuantumVariationalPC(n_qubits=n_qubits, n_layers=2,
                                           eta=0.15, n_iter=100, rng=self.rng))

    def _generator_for_layer(self, l: int):
        """Generator at layer l: given latent z (int → upper-layer
        one-hot), return the predicted lower-layer mean state."""
        W = self.Ws[l]
        n_above = self.layer_sizes[l + 1]
        def gen(z: int) -> np.ndarray:
            return W[:, z % n_above]
        return gen

    def infer(self, x: np.ndarray) -> dict:
        """Run hierarchical inference: sensory x → r_1, r_2, ... posteriors.

        Returns dict with:
          - 'posteriors':  list of per-layer posterior arrays p(z_l) of
            size 2^n_qubits_l.
          - 'maps':        list of per-layer MAP estimates (int).
          - 'free_energies': list of final F_l per layer.
        """
        posteriors = []
        maps = []
        Fs = []
        current = x
        for l in range(len(self.Ws)):
            gen = self._generator_for_layer(l)
            qpc = self.qpc_layers[l]
            # Reset to a fresh near-zero theta so each call starts from
            # a flat prior (previous-call theta would prejudice the result).
            qpc.theta = self.rng.uniform(-0.1, 0.1, size=qpc.ansatz.n_params)
            qpc.fit(gen, current)
            post = qpc.posterior_probs()
            map_z = int(np.argmax(post))
            posteriors.append(post)
            maps.append(map_z)
            Fs.append(qpc.free_energy())
            # Pass the MAP-explanation upward as the next layer's "input":
            # we reconstruct what the upper layer's latent "looks like"
            # in its own coordinate system as a one-hot vector, padded.
            n_above = self.layer_sizes[l + 1]
            one_hot = np.zeros(n_above)
            one_hot[map_z % n_above] = 1.0
            current = one_hot
        return {"posteriors": posteriors, "maps": maps, "free_energies": Fs}

    def predict_top_down(self, top_latent: int) -> list:
        """Generate the predicted activity at every layer from a top-level
        latent index. Returns list of layer activities, lowest first."""
        activities = []
        # Top one-hot.
        n_top = self.layer_sizes[-1]
        a = np.zeros(n_top)
        a[top_latent % n_top] = 1.0
        # Walk down.
        for l in reversed(range(len(self.Ws))):
            a = self.Ws[l] @ a
            activities.append(a.copy())
        return list(reversed(activities))
