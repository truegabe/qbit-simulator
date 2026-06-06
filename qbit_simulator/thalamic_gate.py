"""Thalamic gate -- attention-controlled signal router.

The thalamus is the central relay of the brain.  Every sensory pathway
and most cortico-cortical loops pass through a dedicated thalamic nucleus.
Its job is NOT to compute -- it is to decide what gets through.

Two mechanisms modelled here:

  1. First-order relay (driver input):
     Strong "driver" axons (layer 6 corticothalamic feedback or direct
     sensory input) set WHAT the thalamus relays.

  2. Modulatory gating (attention / arousal):
     A separate "context" signal (attention vector, arousal scalar)
     controls HOW MUCH of the driver gets through.
     Gate value g in [0, 1]:
       g = 0  ->  thalamus closed  (sleep, inattention)
       g = 1  ->  thalamus open    (focused attention)

The gate weights are learned online via a Hebbian-like rule:
  w <- w + lr * (relevance - w) * context
where `relevance` is a task-provided signal (reward, saliency, error).

Classes
-------
  ThalamicGate          -- single nucleus: one input region, one output
  ThalamicRelay         -- multi-nucleus: routes N regions with shared context
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ThalamicGate:
    """Single thalamic nucleus.

    Parameters
    ----------
    n_dims          : dimensionality of the signal passing through
    context_dims    : dimensionality of the attention/arousal context vector
    gate_mode       : 'soft' (continuous [0,1]) or 'hard' (binary threshold)
    threshold       : for hard mode -- gate opens when g > threshold
    lr              : learning rate for gate-weight update
    init_open       : if True, gate starts fully open (weights = 1)
    """
    n_dims:       int
    context_dims: int   = 1        # scalar arousal by default
    gate_mode:    str   = "soft"   # 'soft' | 'hard'
    threshold:    float = 0.5
    lr:           float = 0.05
    init_open:    bool  = True
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    # Learned weights: context -> per-dimension gate value.
    W_gate: np.ndarray = field(default=None, repr=False)  # (n_dims, context_dims)
    _history: list     = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.W_gate is None:
            if self.init_open:
                self.W_gate = np.ones((self.n_dims, self.context_dims))
            else:
                self.W_gate = self.rng.uniform(0, 0.5,
                              (self.n_dims, self.context_dims))

    # ---- gate value ----

    def gate_values(self, context: np.ndarray) -> np.ndarray:
        """Compute per-dimension gate values g in [0,1] from context."""
        c = np.asarray(context, dtype=np.float64).ravel()
        if len(c) != self.context_dims:
            raise ValueError(f"context must have {self.context_dims} dims, "
                             f"got {len(c)}")
        g = self.W_gate @ c                         # (n_dims,)
        g = np.clip(g / (np.abs(g).max() + 1e-9), 0, 1)  # normalize to [0,1]
        return g

    # ---- forward pass ----

    def forward(self, signal: np.ndarray,
                context: np.ndarray) -> tuple[np.ndarray, dict]:
        """Gate the signal using context.

        Returns
        -------
        gated_signal : np.ndarray (n_dims,)
        info         : dict with gate_values, mean_gate, mode
        """
        x = np.asarray(signal, dtype=np.float64).ravel()
        g = self.gate_values(context)
        if self.gate_mode == "hard":
            g_applied = (g > self.threshold).astype(np.float64)
        else:
            g_applied = g
        out = x * g_applied
        info = {
            "gate_values": g,
            "mean_gate":   float(g.mean()),
            "open_fraction": float((g > self.threshold).mean()),
            "mode": self.gate_mode,
        }
        self._history.append(info["mean_gate"])
        return out, info

    # ---- learning ----

    def update(self, context: np.ndarray, relevance: float) -> None:
        """Hebbian gate-weight update.

        relevance in [0, 1]: how important was the signal that just passed?
        (e.g. reward signal, prediction error magnitude, saliency score)
        """
        c = np.asarray(context, dtype=np.float64).ravel()
        # Each weight column pulled toward relevance * context.
        target = relevance * np.ones(self.n_dims)
        current = self.gate_values(c)
        delta = self.lr * (target - current)
        self.W_gate += np.outer(delta, c)
        self.W_gate = np.clip(self.W_gate, 0, None)

    # ---- convenience ----

    def open_all(self) -> None:
        """Force gate fully open (arousal = maximum attention)."""
        self.W_gate[:] = 1.0

    def close_all(self) -> None:
        """Force gate fully closed (sleep / total inattention)."""
        self.W_gate[:] = 0.0

    def mean_openness(self) -> float:
        """Average gate openness over history."""
        return float(np.mean(self._history)) if self._history else 0.0


@dataclass
class ThalamicRelay:
    """Multi-nucleus thalamic relay.

    Routes signals from N input regions to N output regions, each through
    its own ThalamicGate.  A shared context signal (e.g. global arousal +
    task-specific attention vector) modulates all nuclei simultaneously.

    Parameters
    ----------
    n_regions    : number of region pairs to relay
    n_dims_list  : list of signal dimensionalities per region
                   (single int -> same for all)
    context_dims : shared context dimensionality
    """
    n_regions:    int
    n_dims_list:  list | int = 64
    context_dims: int        = 4    # [arousal, valence, novelty, task_relevance]
    gate_mode:    str        = "soft"
    lr:           float      = 0.05
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    nuclei: list = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.n_dims_list, int):
            dims = [self.n_dims_list] * self.n_regions
        else:
            dims = list(self.n_dims_list)
        if not self.nuclei:
            for d in dims:
                self.nuclei.append(ThalamicGate(
                    n_dims=d, context_dims=self.context_dims,
                    gate_mode=self.gate_mode, lr=self.lr, rng=self.rng))

    def route(self, signals: list[np.ndarray],
              context: np.ndarray) -> tuple[list[np.ndarray], list[dict]]:
        """Route all signals through their respective nuclei.

        Returns list of gated signals and list of per-nucleus info dicts.
        """
        outputs, infos = [], []
        for i, (sig, nucleus) in enumerate(zip(signals, self.nuclei)):
            out, info = nucleus.forward(sig, context)
            outputs.append(out)
            infos.append(info)
        return outputs, infos

    def update_all(self, context: np.ndarray,
                   relevances: list[float]) -> None:
        """Update all nucleus gate weights with per-region relevance scores."""
        for nucleus, rel in zip(self.nuclei, relevances):
            nucleus.update(context, rel)

    def mean_openness(self) -> list[float]:
        return [n.mean_openness() for n in self.nuclei]
