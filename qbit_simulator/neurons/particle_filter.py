"""Particle filter for state estimation.

Bayesian state estimation via Monte Carlo: maintain a set of weighted
"particles" approximating the posterior p(x_t | y_{1:t}).

Algorithm at each time step:
  1. Propose: x_t^k ~ p(x_t | x_{t-1}^k)        (dynamics)
  2. Weight: w_t^k ∝ w_{t-1}^k · p(y_t | x_t^k)  (likelihood)
  3. Resample (if needed) to avoid weight degeneracy.

This is the neural-circuit hypothesis for how brains do sequential
Bayesian inference (e.g. Vul et al., Stocker & Simoncelli).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ParticleFilter:
    n_particles: int = 200
    state_dim: int = 1
    particles: np.ndarray = field(default=None, repr=False)
    weights: np.ndarray = field(default=None, repr=False)
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    def __post_init__(self) -> None:
        if self.particles is None:
            self.particles = self.rng.standard_normal((self.n_particles, self.state_dim))
        if self.weights is None:
            self.weights = np.ones(self.n_particles) / self.n_particles

    def predict(self, transition, *args) -> None:
        """transition(x, ...) returns new state for one particle."""
        for i in range(self.n_particles):
            self.particles[i] = transition(self.particles[i], *args)

    def update(self, observation, likelihood) -> None:
        """likelihood(obs, x) returns p(obs | x)."""
        lk = np.array([likelihood(observation, self.particles[i])
                       for i in range(self.n_particles)])
        self.weights *= lk
        s = self.weights.sum()
        if s > 0:
            self.weights /= s
        else:
            self.weights[:] = 1.0 / self.n_particles

    def effective_sample_size(self) -> float:
        return float(1.0 / (self.weights ** 2).sum())

    def resample_if_needed(self, threshold: float = 0.5) -> None:
        if self.effective_sample_size() < threshold * self.n_particles:
            idx = self.rng.choice(self.n_particles, size=self.n_particles,
                                    p=self.weights)
            self.particles = self.particles[idx]
            self.weights[:] = 1.0 / self.n_particles

    def estimate(self) -> np.ndarray:
        return (self.weights[:, None] * self.particles).sum(axis=0)
