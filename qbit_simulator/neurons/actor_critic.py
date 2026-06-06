"""Actor-critic and REINFORCE policy-gradient agents.

REINFORCE (Williams 1992):
    Run an episode under the current policy π_θ.
    For each step t, return G_t = sum_{k>=t} γ^{k-t} r_k.
    θ <- θ + α G_t ∇log π(a_t | s_t)

Actor-Critic adds a value baseline V(s) for variance reduction:
    δ_t = r_t + γ V(s_{t+1}) - V(s_t)
    θ <- θ + α δ_t ∇log π(a_t | s_t)
    w <- w + β δ_t ∇V(s_t)

This implementation: tabular softmax actor + tabular V critic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


@dataclass
class REINFORCE:
    n_states: int
    n_actions: int
    alpha: float = 0.05
    gamma: float = 0.95
    theta: np.ndarray = field(default=None, repr=False)
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    def __post_init__(self) -> None:
        if self.theta is None:
            self.theta = np.zeros((self.n_states, self.n_actions))

    def policy(self, s: int) -> np.ndarray:
        return softmax(self.theta[s])

    def act(self, s: int) -> int:
        return int(self.rng.choice(self.n_actions, p=self.policy(s)))

    def update(self, trajectory: list) -> None:
        """trajectory: list of (s, a, r). Apply REINFORCE update."""
        T = len(trajectory)
        G = 0.0
        # Compute returns from the back.
        returns = []
        for s, a, r in reversed(trajectory):
            G = r + self.gamma * G
            returns.append(G)
        returns.reverse()
        for (s, a, _), G_t in zip(trajectory, returns):
            p = self.policy(s)
            # ∇ log π(a|s) = e_a - p (one-hot minus probs).
            grad = -p
            grad[a] += 1
            self.theta[s] += self.alpha * G_t * grad


@dataclass
class ActorCritic:
    n_states: int
    n_actions: int
    alpha: float = 0.05      # actor lr
    beta:  float = 0.05      # critic lr
    gamma: float = 0.95
    theta: np.ndarray = field(default=None, repr=False)
    V:     np.ndarray = field(default=None, repr=False)
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    def __post_init__(self) -> None:
        if self.theta is None:
            self.theta = np.zeros((self.n_states, self.n_actions))
        if self.V is None:
            self.V = np.zeros(self.n_states)

    def policy(self, s: int) -> np.ndarray:
        return softmax(self.theta[s])

    def act(self, s: int) -> int:
        return int(self.rng.choice(self.n_actions, p=self.policy(s)))

    def step_update(self, s: int, a: int, r: float, s_next: int,
                     done: bool = False) -> float:
        target = r if done else r + self.gamma * self.V[s_next]
        delta = target - self.V[s]
        self.V[s] += self.beta * delta
        p = self.policy(s)
        grad = -p; grad[a] += 1
        self.theta[s] += self.alpha * delta * grad
        return float(delta)
