"""Metaplasticity — sliding LTP/LTD threshold (BCM generalized).

The brain regulates its own plastic capacity. After productive learning
the threshold for further potentiation rises (preventing runaway); after
chaotic learning the threshold lowers (allowing fresh imprinting).

Classical BCM had a sliding threshold θ_M = E[y²]. We generalize:

  - θ_M slides on a fast time scale (per-synapse metaplastic state).
  - θ_M dictates the LTP/LTD crossover for each synapse independently.
  - "Productivity" of recent learning gates how fast θ_M slides.

Update rule (per-synapse w_i):
    dw_i/dt = η · x_i · y · (y - θ_i)
    dθ_i/dt = (y² - θ_i) / τ_θ
    η_eff   = η · g(productivity)

Productivity g is bounded in [η_min, η_max] and tracks how predictable
the network's recent outputs have been (low variance → productive →
larger η).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MetaplasticNeuron:
    """BCM neuron with per-synapse sliding threshold + adaptive η."""
    n_inputs: int
    eta_base: float = 0.005
    eta_min:  float = 0.001
    eta_max:  float = 0.05
    tau_theta: float = 50.0
    tau_prod:  float = 200.0
    w: np.ndarray = field(default=None, repr=False)
    theta: np.ndarray = field(default=None, repr=False)
    productivity: float = 1.0
    _y_var: float = 0.0
    _y_mean: float = 0.0

    def __post_init__(self) -> None:
        if self.w is None:
            self.w = np.random.default_rng(0).uniform(0.0, 0.1, self.n_inputs)
        if self.theta is None:
            self.theta = np.ones(self.n_inputs)

    def response(self, x: np.ndarray) -> float:
        return float(np.dot(self.w, x))

    def step(self, x: np.ndarray) -> float:
        y = self.response(x)
        # Effective learning rate from productivity.
        # High productivity (low variance) → larger η.
        eta_eff = self.eta_base * np.clip(self.productivity,
                                            self.eta_min / self.eta_base,
                                            self.eta_max / self.eta_base)
        # Per-synapse BCM update with PER-SYNAPSE threshold.
        self.w += eta_eff * y * (y - self.theta) * x
        self.w = np.clip(self.w, 0.0, None)
        # Per-synapse sliding threshold: each θ_i tracks E[y²] but
        # weighted by whether input i was active.
        active = (x > 0.1).astype(np.float64)
        self.theta += active * (y * y - self.theta) / self.tau_theta
        # Update output variance trace (productivity = how predictable y is).
        self._y_mean += (y - self._y_mean) / self.tau_prod
        self._y_var  += ((y - self._y_mean) ** 2 - self._y_var) / self.tau_prod
        # Productivity = exp(-σ²) — high when variance is low.
        self.productivity = float(np.exp(-self._y_var))
        return y

    def train(self, X: np.ndarray, n_iter: int = 5000,
               rng: np.random.Generator | None = None) -> dict:
        rng = rng or np.random.default_rng(0)
        ys = []
        for _ in range(n_iter):
            x = X[rng.integers(0, X.shape[0])]
            ys.append(self.step(x))
        responses = np.array([self.response(x) for x in X])
        return {"weights": self.w.copy(),
                "theta_per_syn": self.theta.copy(),
                "responses": responses,
                "selectivity_idx": int(np.argmax(responses)),
                "final_productivity": self.productivity}
