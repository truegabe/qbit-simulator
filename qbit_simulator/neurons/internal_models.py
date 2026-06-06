"""Forward & inverse internal models (Wolpert, Kawato).

The brain learns two complementary models for sensorimotor control:

  - Forward model:  f(state, action) → predicted next state.
    Used to predict consequences (e.g. of motor commands), enabling
    online estimation and cancellation of self-generated sensations.

  - Inverse model:  g(state, desired_next) → action.
    Used to plan actions that achieve a goal state.

Implementation: small linear models trained on (s, a, s') triples
collected from environment interaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ForwardModel:
    state_dim: int
    action_dim: int
    eta: float = 0.01
    W: np.ndarray = field(default=None, repr=False)
    b: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        d = self.state_dim + self.action_dim
        if self.W is None:
            self.W = np.zeros((self.state_dim, d))
        if self.b is None:
            self.b = np.zeros(self.state_dim)

    def predict(self, s: np.ndarray, a: np.ndarray) -> np.ndarray:
        return self.W @ np.concatenate([s, a]) + self.b

    def update(self, s: np.ndarray, a: np.ndarray, s_next: np.ndarray
                ) -> float:
        pred = self.predict(s, a)
        err = s_next - pred
        z = np.concatenate([s, a])
        self.W += self.eta * np.outer(err, z)
        self.b += self.eta * err
        return float(0.5 * (err @ err))


@dataclass
class InverseModel:
    state_dim: int
    action_dim: int
    eta: float = 0.01
    W: np.ndarray = field(default=None, repr=False)
    b: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        d = 2 * self.state_dim
        if self.W is None:
            self.W = np.zeros((self.action_dim, d))
        if self.b is None:
            self.b = np.zeros(self.action_dim)

    def predict(self, s: np.ndarray, s_next: np.ndarray) -> np.ndarray:
        return self.W @ np.concatenate([s, s_next]) + self.b

    def update(self, s: np.ndarray, s_next: np.ndarray, a: np.ndarray
                ) -> float:
        pred = self.predict(s, s_next)
        err = a - pred
        z = np.concatenate([s, s_next])
        self.W += self.eta * np.outer(err, z)
        self.b += self.eta * err
        return float(0.5 * (err @ err))


@dataclass
class PairedInternalModels:
    """Both models trained together from (s, a, s') experiences."""
    state_dim: int
    action_dim: int
    eta: float = 0.01
    fwd: ForwardModel = field(default=None)
    inv: InverseModel = field(default=None)

    def __post_init__(self) -> None:
        if self.fwd is None:
            self.fwd = ForwardModel(self.state_dim, self.action_dim, self.eta)
        if self.inv is None:
            self.inv = InverseModel(self.state_dim, self.action_dim, self.eta)

    def train_step(self, s, a, s_next) -> tuple[float, float]:
        return self.fwd.update(s, a, s_next), self.inv.update(s, s_next, a)
