"""Options framework / hierarchical RL (Sutton, Precup, Singh 1999).

An "option" is a temporally extended action with:
  - initiation set I (states where it can start)
  - intra-option policy π_o
  - termination condition β_o(s) ∈ [0, 1]

The higher-level policy chooses options; each option runs until it
terminates. This adds temporal abstraction.

Implementation: tabular options with deterministic initiation and
termination, and a meta-policy over options trained by Q-learning over
"semi-Markov" decision steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Option:
    name: str
    initiation: set            # states where this option can begin
    policy: dict               # state -> action
    termination: set           # states where the option terminates


@dataclass
class OptionsAgent:
    """Hierarchical agent: meta-policy over options + intra-policies fixed."""
    n_states: int
    options: list
    alpha: float = 0.1
    gamma: float = 0.95
    eps: float = 0.1
    Q: np.ndarray = field(default=None, repr=False)
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    def __post_init__(self) -> None:
        if self.Q is None:
            self.Q = np.zeros((self.n_states, len(self.options)))

    def available_options(self, s: int) -> list:
        return [i for i, o in enumerate(self.options) if s in o.initiation]

    def select_option(self, s: int) -> int:
        avail = self.available_options(s)
        if not avail:
            return -1
        if self.rng.uniform() < self.eps:
            return int(self.rng.choice(avail))
        # Argmax over available.
        qs = self.Q[s, avail]
        return avail[int(np.argmax(qs))]

    def run_option(self, s: int, opt_idx: int, env_step) -> tuple[int, float, int]:
        """Execute option opt_idx from state s until it terminates.

        env_step(s, a) -> (s_next, r, done).
        Returns: (s_terminal, cumulative_discounted_reward, k_steps).
        """
        o = self.options[opt_idx]
        cum = 0.0; k = 0
        gamma_k = 1.0
        while True:
            a = o.policy.get(s, 0)
            s_next, r, done = env_step(s, a)
            cum += gamma_k * r
            gamma_k *= self.gamma
            k += 1
            s = s_next
            if done or s in o.termination:
                return s, cum, k

    def update(self, s: int, opt_idx: int, cum_r: float,
                s_term: int, k: int, done: bool = False) -> None:
        if done or not self.available_options(s_term):
            target = cum_r
        else:
            next_avail = self.available_options(s_term)
            target = cum_r + (self.gamma ** k) * np.max(self.Q[s_term, next_avail])
        self.Q[s, opt_idx] += self.alpha * (target - self.Q[s, opt_idx])
