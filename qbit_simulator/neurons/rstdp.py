"""Reward-modulated STDP (R-STDP).

Vanilla STDP is HEBBIAN — it strengthens correlations regardless of
whether they're useful. R-STDP (Florian 2007; Izhikevich 2007) adds a
global "reward" signal R(t) that gates plasticity, turning the local
learning rule into a global credit-assignment mechanism:

    Δw_ij(t)  =  R(t) · eligibility_trace_ij(t)

where the eligibility trace tracks "would-have-been STDP" updates but
defers them. When a reward arrives, the eligibility traces are CONVERTED
into actual weight changes, scaled by R.

This solves the XOR problem and is essentially a biologically-plausible
form of policy gradient. It's also Izhikevich's "dopamine-modulated
STDP" — the dopamine pulse from the basal ganglia is the global R(t).

Eligibility-trace dynamics (per synapse):

    dE_ij/dt = -E_ij / tau_e   (decay)
    On STDP event Δw_STDP at synapse ij:
        E_ij += Δw_STDP

    On reward R(t):
        w_ij += R(t) · E_ij · dt

Provides:
  - `EligibilityTrace`: trace + decay + STDP integration.
  - `RSTDPLearner(n_pre, n_post, ...)`: a fused STDP-trace + reward-gated
    weight updater.
  - `train_with_rstdp(...)`: full driver — accepts a reward schedule and
    pre/post spike trains, updates weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .stdp import STDPRule


# ----------------------------------------------------------------------------
# Eligibility trace + R-STDP learner
# ----------------------------------------------------------------------------

@dataclass
class RSTDPLearner:
    """Fused R-STDP weight-update mechanism.

    State:
      - `pre_trace`, `post_trace`:  STDP-style traces (one per pre/post neuron).
      - `eligibility`:              per-synapse trace of pending Δw values,
                                    shape (n_post, n_pre).

    Each step:
      1. Decay all three traces.
      2. On pre/post spikes, update STDP traces and ADD their immediate
         contribution to `eligibility`.
      3. If R(t) ≠ 0, apply `weights += lr · R(t) · eligibility`.
    """
    n_pre:  int
    n_post: int
    rule:   STDPRule
    tau_eligibility: float = 200.0   # how long credit lingers (ms-ish)
    lr:     float = 0.01

    pre_trace:   np.ndarray = field(default=None, repr=False)
    post_trace:  np.ndarray = field(default=None, repr=False)
    eligibility: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.pre_trace is None:
            self.pre_trace = np.zeros(self.n_pre)
        if self.post_trace is None:
            self.post_trace = np.zeros(self.n_post)
        if self.eligibility is None:
            self.eligibility = np.zeros((self.n_post, self.n_pre))

    def reset(self) -> None:
        self.pre_trace[:] = 0
        self.post_trace[:] = 0
        self.eligibility[:] = 0

    def step(
        self, weights: np.ndarray,
        pre_spikes: np.ndarray, post_spikes: np.ndarray,
        reward: float, dt: float = 1.0,
    ) -> np.ndarray:
        """One time step.

        Args:
            weights:     shape (n_post, n_pre) — modified in-place style.
            pre_spikes:  bool array shape (n_pre,).
            post_spikes: bool array shape (n_post,).
            reward:      scalar dopamine-like signal at this step.
            dt:          time step.

        Returns:
            updated weights (also stored internally via class state).
        """
        # 1. Decay traces.
        self.pre_trace  *= np.exp(-dt / self.rule.tau_plus)
        self.post_trace *= np.exp(-dt / self.rule.tau_minus)
        self.eligibility *= np.exp(-dt / self.tau_eligibility)

        # 2. On pre spike: contribute DEPRESSION = -A_- * post_trace
        #    to eligibility[i, j] for each post i, pre j.
        if pre_spikes.any():
            depression = np.outer(self.post_trace, pre_spikes.astype(float))
            self.eligibility = self.eligibility - self.rule.A_minus * depression
            self.pre_trace = self.pre_trace + pre_spikes.astype(float)

        # 3. On post spike: contribute POTENTIATION = +A_+ * pre_trace
        if post_spikes.any():
            potentiation = np.outer(post_spikes.astype(float), self.pre_trace)
            self.eligibility = self.eligibility + self.rule.A_plus * potentiation
            self.post_trace = self.post_trace + post_spikes.astype(float)

        # 4. Reward-modulated weight update.
        if reward != 0.0:
            weights = weights + self.lr * reward * self.eligibility * dt
            weights = self.rule.clip(weights)

        return weights


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def train_with_rstdp(
    weights: np.ndarray,
    pre_spike_history: np.ndarray,
    post_spike_history: np.ndarray,
    reward_schedule: np.ndarray,
    rule: STDPRule | None = None,
    tau_eligibility: float = 200.0,
    lr: float = 0.01,
) -> tuple[np.ndarray, dict]:
    """Apply R-STDP to a weight matrix given full spike + reward histories.

    Args:
        weights:          shape (n_post, n_pre).
        pre_spike_history:  shape (n_steps, n_pre), bool.
        post_spike_history: shape (n_steps, n_post), bool.
        reward_schedule:    shape (n_steps,), scalar R(t) at each step.
        rule:               STDP rule (default: built-in).
        tau_eligibility:    eligibility-trace decay constant.
        lr:                 learning rate.

    Returns:
        (updated_weights, diagnostics).
    """
    if rule is None:
        rule = STDPRule()
    n_post, n_pre = weights.shape
    n_steps = pre_spike_history.shape[0]
    if (pre_spike_history.shape != (n_steps, n_pre)
            or post_spike_history.shape != (n_steps, n_post)
            or reward_schedule.shape != (n_steps,)):
        raise ValueError("spike/reward shapes inconsistent with weights")

    learner = RSTDPLearner(
        n_pre=n_pre, n_post=n_post,
        rule=rule, tau_eligibility=tau_eligibility, lr=lr,
    )
    weight_history = []
    eligibility_norm_history = []
    for t in range(n_steps):
        pre = pre_spike_history[t].astype(bool)
        post = post_spike_history[t].astype(bool)
        weights = learner.step(weights, pre, post,
                                reward=float(reward_schedule[t]))
        weight_history.append(weights.copy())
        eligibility_norm_history.append(float(np.linalg.norm(learner.eligibility)))

    return weights, {
        "weight_history":              weight_history,
        "eligibility_norm_history":    eligibility_norm_history,
        "final_eligibility":           learner.eligibility.copy(),
        "n_steps":                     n_steps,
    }


# ----------------------------------------------------------------------------
# Reward schedules
# ----------------------------------------------------------------------------

def pulse_reward(n_steps: int, pulse_times: list[int],
                   amplitude: float = 1.0) -> np.ndarray:
    """Delta-pulse reward: zero everywhere, nonzero at given time steps."""
    out = np.zeros(n_steps)
    for t in pulse_times:
        if 0 <= t < n_steps:
            out[t] = amplitude
    return out


def exponential_reward(n_steps: int, pulse_times: list[int],
                          amplitude: float = 1.0, tau: float = 10.0
                          ) -> np.ndarray:
    """Pulse of reward with exponential tail (more realistic dopamine pulse)."""
    out = np.zeros(n_steps)
    for t_pulse in pulse_times:
        for t in range(t_pulse, min(n_steps, t_pulse + int(5 * tau))):
            out[t] += amplitude * np.exp(-(t - t_pulse) / tau)
    return out
