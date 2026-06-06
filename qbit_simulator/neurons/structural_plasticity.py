"""Structural plasticity — synapse creation and deletion.

Biological synapses are not fixed: weak synapses retract over time and
new synapses sprout from active axons toward active dendrites. This
provides a fundamentally different form of long-term adaptation than
weight tuning.

This module wraps a (n_post × n_pre) weight matrix with two ongoing
processes:

  - PRUNING: synapses with |w| below a sliding threshold are deleted
    (set to 0).
  - GROWTH:  with some probability, NEW synapses sprout between a
    randomly-chosen post/pre pair that is currently disconnected.
    Probability is biased by recent co-activation
    (Hebb-like "use it or lose it").

Activity statistics are maintained as exponentially-decayed traces.
The structural changes happen on a slow time scale; weight changes
(STDP, BCM, etc.) happen faster and are unaffected here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StructuralPlasticityManager:
    """Manages structural changes to a weight matrix.

    Maintains:
      - `active_trace_pre`, `active_trace_post`: decayed co-activity.
      - `coact_trace`: trace of pre-post coincidences per synapse.
    """
    n_pre: int
    n_post: int
    tau_act: float = 100.0       # activity-trace decay
    tau_coact: float = 200.0     # coactivation-trace decay
    prune_threshold: float = 0.01
    prune_rate: float = 0.05     # fraction of weak synapses pruned per call
    growth_rate: float = 0.02    # prob of growth per active (post, pre) per call
    init_weight: float = 0.1
    max_density: float = 0.5     # cap on connection density
    active_trace_pre:  np.ndarray = field(default=None, repr=False)
    active_trace_post: np.ndarray = field(default=None, repr=False)
    coact_trace:       np.ndarray = field(default=None, repr=False)
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    def __post_init__(self) -> None:
        if self.active_trace_pre is None:
            self.active_trace_pre = np.zeros(self.n_pre)
        if self.active_trace_post is None:
            self.active_trace_post = np.zeros(self.n_post)
        if self.coact_trace is None:
            self.coact_trace = np.zeros((self.n_post, self.n_pre))

    def observe_activity(self, pre_act: np.ndarray, post_act: np.ndarray,
                          dt: float = 1.0) -> None:
        """Update activity traces."""
        decay_act = np.exp(-dt / self.tau_act)
        decay_co  = np.exp(-dt / self.tau_coact)
        self.active_trace_pre  = decay_act * self.active_trace_pre + pre_act
        self.active_trace_post = decay_act * self.active_trace_post + post_act
        self.coact_trace = decay_co * self.coact_trace \
                            + np.outer(post_act, pre_act)

    def prune(self, W: np.ndarray) -> tuple[np.ndarray, int]:
        """Delete weak synapses. Returns updated W and number deleted."""
        # Candidate-pruning mask: connected and weak.
        connected = (W != 0)
        weak = (np.abs(W) < self.prune_threshold) & connected
        # Stochastic pruning of weak ones.
        rand = self.rng.uniform(size=W.shape)
        prune_mask = weak & (rand < self.prune_rate)
        n_pruned = int(prune_mask.sum())
        W = W.copy()
        W[prune_mask] = 0
        return W, n_pruned

    def grow(self, W: np.ndarray) -> tuple[np.ndarray, int]:
        """Add new synapses where pre and post are co-active."""
        density = float((W != 0).mean())
        if density >= self.max_density:
            return W, 0
        # Co-activation pressure normalized.
        coact = self.coact_trace.copy()
        if coact.max() > 0:
            coact /= coact.max()
        # Probability per (post, pre): only where disconnected.
        disconnected = (W == 0)
        prob = self.growth_rate * coact * disconnected
        rand = self.rng.uniform(size=W.shape)
        grow_mask = rand < prob
        n_grown = int(grow_mask.sum())
        W = W.copy()
        # New synapse: small positive weight, sign chosen randomly.
        signs = self.rng.choice([-1, 1], size=W.shape)
        W = np.where(grow_mask, signs * self.init_weight, W)
        return W, n_grown

    def step(self, W: np.ndarray,
              pre_act: np.ndarray | None = None,
              post_act: np.ndarray | None = None,
              dt: float = 1.0) -> dict:
        """One step of structural update.

        Returns dict with new W, n_pruned, n_grown, density.
        """
        if pre_act is not None and post_act is not None:
            self.observe_activity(pre_act, post_act, dt=dt)
        W, n_p = self.prune(W)
        W, n_g = self.grow(W)
        density = float((W != 0).mean())
        return {"W": W, "n_pruned": n_p, "n_grown": n_g, "density": density}
