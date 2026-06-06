"""Spike-Timing-Dependent Plasticity (STDP).

STDP is the experimentally-observed Hebbian learning rule:

  - If a PRE-synaptic spike arrives BEFORE a POST-synaptic spike (i.e.
    pre → post causal order), the synapse is POTENTIATED (Δw > 0).
  - If POST fires BEFORE PRE, the synapse is DEPRESSED (Δw < 0).

The classic "asymmetric exponential STDP window" (Bi-Poo 1998):

    Δw(Δt) =  +A_+ · exp(-Δt / tau_+)   if Δt > 0  (pre before post)
              -A_- · exp(+Δt / tau_-)   if Δt < 0  (post before pre)

where Δt = t_post − t_pre. Typical values: A_+ ≈ A_- ≈ 0.01,
tau_+ ≈ tau_- ≈ 20 ms.

We implement this both as:
  - A static rule: `stdp_weight_change(delta_t, ...)`.
  - A trace-based online rule: each neuron carries pre- and post-
    synaptic eligibility traces that decay exponentially; on each spike,
    weights update based on the current trace value.

The trace-based version is computationally efficient (O(N) per step
instead of O(N · spike history length)) and is what we use in SNN
simulations.

This module also provides:
  - Weight bounds (no negative or unbounded weights).
  - A configurable STDP rule object.
  - `train_with_stdp(snn, input_pattern, n_steps, ...)` for one-shot
    use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------------------------
# STDP rule (parameters + static formula)
# ----------------------------------------------------------------------------

@dataclass
class STDPRule:
    """Asymmetric exponential STDP rule (Bi-Poo)."""
    A_plus:  float = 0.01      # potentiation amplitude
    A_minus: float = 0.012     # depression amplitude (often slightly > A_+)
    tau_plus:  float = 20.0    # potentiation time constant
    tau_minus: float = 20.0    # depression time constant
    w_min:   float = 0.0       # lower bound on weight
    w_max:   float = 1.0       # upper bound

    def delta_w(self, delta_t: float) -> float:
        """STDP curve: Δw as a function of Δt = t_post − t_pre."""
        if delta_t > 0:
            return self.A_plus * np.exp(-delta_t / self.tau_plus)
        elif delta_t < 0:
            return -self.A_minus * np.exp(delta_t / self.tau_minus)
        return 0.0

    def clip(self, w: np.ndarray | float) -> np.ndarray | float:
        """Clip weight to [w_min, w_max]."""
        return np.clip(w, self.w_min, self.w_max)


# ----------------------------------------------------------------------------
# Trace-based online STDP (the efficient version)
# ----------------------------------------------------------------------------

@dataclass
class STDPTraces:
    """Eligibility traces for trace-based STDP.

    Each pre-synaptic neuron has a "pre trace" that jumps up by 1 on
    each spike and decays exponentially with tau_plus. Similarly each
    post-synaptic neuron has a "post trace" with tau_minus.

    When neuron j (pre) spikes: w_ij += A_minus · post_trace_i  for
    all post neurons i (depression: a recent post spike has earned
    the synapse a debit).

    When neuron i (post) spikes: w_ij += A_plus · pre_trace_j  for
    all pre neurons j (potentiation: a recent pre spike earns credit).

    This trace-based formulation is mathematically equivalent to summing
    the static rule over all spike-pair timings.
    """
    n_pre:    int
    n_post:   int
    rule:     STDPRule

    pre_trace:  np.ndarray | None = None    # shape (n_pre,)
    post_trace: np.ndarray | None = None    # shape (n_post,)

    def __post_init__(self) -> None:
        if self.pre_trace is None:
            self.pre_trace = np.zeros(self.n_pre)
        if self.post_trace is None:
            self.post_trace = np.zeros(self.n_post)

    def step_decay(self, dt: float = 1.0) -> None:
        """Decay both traces by one time step."""
        self.pre_trace  *= np.exp(-dt / self.rule.tau_plus)
        self.post_trace *= np.exp(-dt / self.rule.tau_minus)

    def update_weights_on_pre_spike(
        self, weights: np.ndarray, pre_spikes: np.ndarray,
    ) -> np.ndarray:
        """Apply DEPRESSION for any pre spikes: w_ij -= A_- · post_trace_i.

        weights has shape (n_post, n_pre).
        pre_spikes is a boolean array of shape (n_pre,).
        """
        if not pre_spikes.any():
            return weights
        # For each pre spike, subtract A_- * post_trace from the
        # corresponding column.
        delta = np.outer(self.post_trace, pre_spikes.astype(float))
        weights = weights - self.rule.A_minus * delta
        weights = self.rule.clip(weights)
        # Increment the pre trace where pre spiked.
        self.pre_trace = self.pre_trace + pre_spikes.astype(float)
        return weights

    def update_weights_on_post_spike(
        self, weights: np.ndarray, post_spikes: np.ndarray,
    ) -> np.ndarray:
        """Apply POTENTIATION for any post spikes:
        w_ij += A_+ · pre_trace_j  for each spiking post neuron i.
        """
        if not post_spikes.any():
            return weights
        delta = np.outer(post_spikes.astype(float), self.pre_trace)
        weights = weights + self.rule.A_plus * delta
        weights = self.rule.clip(weights)
        self.post_trace = self.post_trace + post_spikes.astype(float)
        return weights


# ----------------------------------------------------------------------------
# Pairwise (offline) STDP — for verification + small examples
# ----------------------------------------------------------------------------

def pairwise_stdp_update(
    pre_spike_times: np.ndarray, post_spike_times: np.ndarray,
    rule: STDPRule,
) -> float:
    """Compute the cumulative Δw for a single synapse over the
    cross-product of pre- and post-spike times.

    Equivalent to the "all-to-all" interaction model. Useful as a
    reference for the trace-based version.
    """
    total = 0.0
    for t_pre in pre_spike_times:
        for t_post in post_spike_times:
            total += rule.delta_w(t_post - t_pre)
    return float(total)


# ----------------------------------------------------------------------------
# Training driver
# ----------------------------------------------------------------------------

def train_with_stdp(
    weights: np.ndarray,
    pre_spike_history: np.ndarray,
    post_spike_history: np.ndarray,
    rule: STDPRule,
) -> tuple[np.ndarray, dict]:
    """Apply STDP updates to a weight matrix based on observed spikes.

    Args:
        weights: shape (n_post, n_pre).
        pre_spike_history:  shape (n_steps, n_pre), bool.
        post_spike_history: shape (n_steps, n_post), bool.
        rule:    STDPRule.

    Returns:
        (updated_weights, diagnostics).
    """
    n_post, n_pre = weights.shape
    if pre_spike_history.shape[1] != n_pre:
        raise ValueError(
            f"pre history n_pre {pre_spike_history.shape[1]} != weights {n_pre}"
        )
    if post_spike_history.shape[1] != n_post:
        raise ValueError(
            f"post history n_post {post_spike_history.shape[1]} != weights {n_post}"
        )
    n_steps = pre_spike_history.shape[0]

    traces = STDPTraces(n_pre=n_pre, n_post=n_post, rule=rule)
    history = []
    for t in range(n_steps):
        traces.step_decay()
        pre = pre_spike_history[t].astype(bool)
        post = post_spike_history[t].astype(bool)
        # CONVENTION: pre-spike update uses CURRENT post trace BEFORE
        # any post-spike arrives this step. Then post-spike update.
        weights = traces.update_weights_on_pre_spike(weights, pre)
        weights = traces.update_weights_on_post_spike(weights, post)
        history.append(weights.copy())
    return weights, {
        "weight_history": history,
        "final_weights":  weights,
        "n_steps":        n_steps,
    }
