"""Cortical microcircuit: Mountcastle's 6-layer canonical column.

The canonical cortical column (Mountcastle 1957; Douglas-Martin 2004)
is the repeating unit of mammalian neocortex: a vertical "stack" of 6
layers, each with stereotyped cell types and inter-layer connectivity:

    L1: sparse interneurons + apical dendrites
    L2/3: superficial pyramidal cells (cortico-cortical projection)
    L4: granular layer (thalamic input target)
    L5: deep pyramidal cells (subcortical output)
    L6: thalamic feedback

Canonical microcircuit information flow:

    Thalamic input → L4 → L2/3 → L5 (→ output) ↘
                                        ↓        L6 → thalamus (feedback)
                                       L2/3 ↗   feedback loop
                                       L1 ← apical dendrites

This module implements a simplified spiking version:
  - Each layer is a small LIFPopulation.
  - Inter-layer weights are tuned to reproduce the canonical
    information flow.
  - A `CorticalColumn(n_per_layer)` object exposes `step(thalamic_input)`
    and `run(...)`.
  - Output is read from L5 (the cortical-output layer).

Two columns can be connected via lateral (L2/3 → L2/3) and feedback
(L5 → L1) projections.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .lif import LIFPopulation


LAYER_NAMES = ["L1", "L23", "L4", "L5", "L6"]


@dataclass
class CorticalColumn:
    """One canonical cortical column with 5 effective layers (we merge
    L2/3 since they're functionally one layer).

    Per-layer connectivity (canonical microcircuit):
      - thalamic input → L4 (excitatory drive)
      - L4 → L2/3 (excitatory, strong)
      - L2/3 → L5 (excitatory)
      - L5 → L6 (excitatory)
      - L6 → L4 (excitatory, "feedback amplification")
      - L1 → L2/3 apical dendrites (excitatory modulatory)
    """
    n_per_layer: int = 20
    syn_tau: float = 25.0          # synaptic decay
    w_drive: float = 1.5           # input-to-L4 weight
    w_L4_to_L23: float = 1.0
    w_L23_to_L5: float = 1.0
    w_L5_to_L6: float = 0.6
    w_L6_to_L4: float = 0.4
    w_L1_to_L23: float = 0.5
    tonic: float = 0.0

    layers:        dict = field(default_factory=dict)
    syn_currents:  dict = field(default_factory=dict)
    spike_history: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in LAYER_NAMES:
            self.layers[name] = LIFPopulation(n=self.n_per_layer)
            self.syn_currents[name] = np.zeros(self.n_per_layer)
            self.spike_history[name] = []

    def reset(self) -> None:
        for name in LAYER_NAMES:
            self.layers[name].reset()
            self.syn_currents[name][:] = 0
            self.spike_history[name] = []

    def _last_spikes(self, name: str) -> np.ndarray:
        h = self.spike_history[name]
        return h[-1] if h else np.zeros(self.n_per_layer, dtype=bool)

    def step(self, thalamic_input: float = 0.0,
              top_down_input: float = 0.0, t: int = 0) -> dict:
        """One time step of the column.

        Args:
            thalamic_input: drive into L4 (bottom-up).
            top_down_input: drive into L1 (top-down modulation).
            t:              time index.

        Returns:
            dict mapping layer name → spike vector.
        """
        # Decay synaptic currents.
        decay = np.exp(-1.0 / self.syn_tau)
        for name in LAYER_NAMES:
            self.syn_currents[name] *= decay

        # Apply external drives.
        L4_in = self.w_drive * thalamic_input
        L1_in = self.w_drive * top_down_input

        # Add cross-layer contributions from PREVIOUS spike timestep.
        L4_spikes = self._last_spikes("L4")
        L23_spikes = self._last_spikes("L23")
        L5_spikes = self._last_spikes("L5")
        L6_spikes = self._last_spikes("L6")
        L1_spikes = self._last_spikes("L1")

        self.syn_currents["L4"]  += self.w_L6_to_L4 * L6_spikes.mean() * 5
        self.syn_currents["L23"] += self.w_L4_to_L23 * L4_spikes.mean() * 5
        self.syn_currents["L23"] += self.w_L1_to_L23 * L1_spikes.mean() * 3
        self.syn_currents["L5"]  += self.w_L23_to_L5 * L23_spikes.mean() * 5
        self.syn_currents["L6"]  += self.w_L5_to_L6 * L5_spikes.mean() * 5

        # External drives apply per-step.
        drives = {
            "L1": np.full(self.n_per_layer, self.tonic + L1_in),
            "L23": np.full(self.n_per_layer, self.tonic),
            "L4": np.full(self.n_per_layer, self.tonic + L4_in),
            "L5": np.full(self.n_per_layer, self.tonic),
            "L6": np.full(self.n_per_layer, self.tonic),
        }
        spikes = {}
        for name in LAYER_NAMES:
            total_input = drives[name] + self.syn_currents[name]
            s = self.layers[name].step(total_input, t=t)
            self.spike_history[name].append(s.copy())
            spikes[name] = s
        return spikes

    def run(self, thalamic_input: float, n_steps: int = 100,
              top_down_input: float = 0.0) -> dict:
        """Run for n_steps with constant inputs."""
        self.reset()
        for t in range(n_steps):
            self.step(thalamic_input, top_down_input, t=t)
        return {
            "rates": {name: float(np.mean(self.spike_history[name]))
                       for name in LAYER_NAMES},
            "spike_history": {name: np.array(self.spike_history[name])
                               for name in LAYER_NAMES},
        }


# ----------------------------------------------------------------------------
# Multi-column "patch" of cortex
# ----------------------------------------------------------------------------

@dataclass
class CorticalPatch:
    """A small lattice of cortical columns with lateral L2/3 connections."""
    n_columns: int = 4
    n_per_layer: int = 15
    lateral_weight: float = 0.3
    columns: list[CorticalColumn] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.columns:
            self.columns = [
                CorticalColumn(n_per_layer=self.n_per_layer)
                for _ in range(self.n_columns)
            ]

    def reset(self) -> None:
        for c in self.columns:
            c.reset()

    def run(self, thalamic_inputs: np.ndarray, n_steps: int = 100) -> dict:
        """Run all columns in parallel; thalamic_inputs has shape (n_columns,)."""
        if len(thalamic_inputs) != self.n_columns:
            raise ValueError(
                f"thalamic_inputs length {len(thalamic_inputs)} != "
                f"n_columns {self.n_columns}"
            )
        self.reset()
        for t in range(n_steps):
            # Each column gets its thalamic drive + lateral input from
            # neighboring columns' L2/3.
            for c_idx, col in enumerate(self.columns):
                lateral = 0.0
                for nbr_idx, nbr_col in enumerate(self.columns):
                    if nbr_idx == c_idx:
                        continue
                    last_L23 = nbr_col._last_spikes("L23")
                    lateral += self.lateral_weight * last_L23.mean()
                col.step(
                    thalamic_input=float(thalamic_inputs[c_idx]) + lateral,
                    t=t,
                )
        return {
            "rates": [
                {name: float(np.mean(col.spike_history[name]))
                 for name in LAYER_NAMES}
                for col in self.columns
            ],
        }
