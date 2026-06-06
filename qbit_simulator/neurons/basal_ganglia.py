"""Basal-ganglia model (action selection).

The basal ganglia (BG) are a set of subcortical nuclei central to
action selection and reinforcement learning. We implement the
canonical "direct/indirect pathway" architecture:

    Cortex → Striatum → GPi → Thalamus → Cortex
         direct (D1):  - inhibits  GPi (releases thalamus)
         indirect (D2): inhibits  GPe → disinhibits STN → excites GPi

Net effect: D1 path PROMOTES selected action; D2 path SUPPRESSES
competitors. Dopamine modulates the gain of these pathways
(D1 facilitates direct, D2 inhibits indirect).

Action selection: winner-take-all over actions where the GPi output is
LOWEST (i.e., thalamus is most disinhibited).

Q-learning interpretation: the cortico-striatal weights are the
state-action values, updated by R-STDP-style three-factor rules using
dopamine as the prediction-error signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BasalGanglia:
    """Action-selection model."""
    n_state: int
    n_actions: int
    alpha: float = 0.1          # learning rate
    gamma: float = 0.9          # RL discount
    dopa_base: float = 0.5      # baseline DA
    # Cortico-striatal weights (action values).
    Q: np.ndarray = field(default=None, repr=False)
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    def __post_init__(self) -> None:
        if self.Q is None:
            self.Q = np.zeros((self.n_state, self.n_actions))

    def gpi_output(self, state: int, dopamine: float) -> np.ndarray:
        """Per-action GPi output (lower = more selected).

        Direct path (D1, gain ∝ dopamine): inhibits GPi proportional to Q.
        Indirect path (D2, gain ∝ 1 - dopamine): provides broad NoGo
        bias that opposes any commitment — implemented as a baseline
        positive contribution proportional to OTHER actions' Q (so good
        competitors raise the GPi for action a too).
        """
        Q = self.Q[state]
        direct = dopamine * Q
        # competitors[a] = sum_{b != a} Q[b]
        competitors = Q.sum() - Q
        indirect = (1 - dopamine) * competitors
        return -direct + indirect

    def select(self, state: int, dopamine: float = None,
                eps: float = 0.1) -> int:
        """Pick action by argmin GPi output, with ε-greedy exploration."""
        if dopamine is None:
            dopamine = self.dopa_base
        if self.rng.uniform() < eps:
            return int(self.rng.integers(0, self.n_actions))
        gpi = self.gpi_output(state, dopamine)
        # Action with most disinhibition = lowest GPi.
        return int(np.argmin(gpi))

    def update(self, state: int, action: int, reward: float,
                next_state: int, done: bool = False) -> float:
        """TD-learning update. Returns the prediction error δ."""
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        delta = target - self.Q[state, action]
        self.Q[state, action] += self.alpha * delta
        return float(delta)

    def greedy_action(self, state: int) -> int:
        return int(np.argmax(self.Q[state]))


def train_bg_on_chain(bg: BasalGanglia, n_states: int, n_episodes: int = 200,
                      max_steps: int = 50) -> list:
    """Simple chain world: reward at the last state. Returns per-episode
    cumulative reward."""
    rewards = []
    for ep in range(n_episodes):
        s = 0
        cum = 0.0
        for t in range(max_steps):
            a = bg.select(s, eps=max(0.5 * (1 - ep / n_episodes), 0.05))
            # Action 0 = left, 1 = right (assuming n_actions==2).
            if a == 1:
                s_next = min(s + 1, n_states - 1)
            else:
                s_next = max(s - 1, 0)
            r = 1.0 if s_next == n_states - 1 else 0.0
            done = s_next == n_states - 1
            bg.update(s, a, r, s_next, done)
            s = s_next
            cum += r
            if done:
                break
        rewards.append(cum)
    return rewards
