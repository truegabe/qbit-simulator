"""Tabular Q-learning, SARSA, Expected-SARSA.

Three classic temporal-difference RL agents on discrete state/action
spaces.

Q-learning (off-policy):
    Q(s, a) <- Q(s, a) + α [r + γ max_a' Q(s', a') - Q(s, a)]

SARSA (on-policy):
    Q(s, a) <- Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]

Expected SARSA (mid-way):
    Q(s, a) <- Q(s, a) + α [r + γ E_π[Q(s', a')] - Q(s, a)]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class QLearning:
    n_states: int
    n_actions: int
    alpha: float = 0.1
    gamma: float = 0.95
    eps: float = 0.1
    Q: np.ndarray = field(default=None, repr=False)
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    def __post_init__(self) -> None:
        if self.Q is None:
            self.Q = np.zeros((self.n_states, self.n_actions))

    def act(self, s: int) -> int:
        if self.rng.uniform() < self.eps:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.Q[s]))

    def update(self, s: int, a: int, r: float, s_next: int,
                done: bool = False) -> float:
        target = r if done else r + self.gamma * np.max(self.Q[s_next])
        delta = target - self.Q[s, a]
        self.Q[s, a] += self.alpha * delta
        return float(delta)


@dataclass
class SARSA(QLearning):
    """On-policy: uses the action actually taken next."""
    def update(self, s: int, a: int, r: float, s_next: int,
                a_next: int, done: bool = False) -> float:
        target = r if done else r + self.gamma * self.Q[s_next, a_next]
        delta = target - self.Q[s, a]
        self.Q[s, a] += self.alpha * delta
        return float(delta)


@dataclass
class ExpectedSARSA(QLearning):
    """Uses expectation over ε-greedy policy of next Q values."""
    def update(self, s: int, a: int, r: float, s_next: int,
                done: bool = False) -> float:
        if done:
            target = r
        else:
            q_next = self.Q[s_next]
            best = np.argmax(q_next)
            # ε-greedy policy probabilities.
            probs = np.full(self.n_actions, self.eps / self.n_actions)
            probs[best] += 1 - self.eps
            expected = float(probs @ q_next)
            target = r + self.gamma * expected
        delta = target - self.Q[s, a]
        self.Q[s, a] += self.alpha * delta
        return float(delta)


def run_episode_grid(agent: QLearning, n_states: int = 5,
                      max_steps: int = 50) -> float:
    """Simple chain: state 0 → n-1 with reward at goal."""
    s = 0; cum = 0.0
    for _ in range(max_steps):
        a = agent.act(s)
        s_next = min(s + 1, n_states - 1) if a == 1 else max(s - 1, 0)
        r = 1.0 if s_next == n_states - 1 else 0.0
        done = s_next == n_states - 1
        if isinstance(agent, SARSA):
            a_next = agent.act(s_next)
            agent.update(s, a, r, s_next, a_next, done)
        else:
            agent.update(s, a, r, s_next, done)
        s = s_next; cum += r
        if done:
            break
    return cum
