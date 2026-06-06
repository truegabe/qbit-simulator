"""Wilson-Cowan equations — population-level E/I dynamics.

The classic rate-level model of cortical populations:

    τ_E dE/dt = -E + (1-E) S(w_EE E - w_EI I + P)
    τ_I dI/dt = -I + (1-I) S(w_IE E - w_II I + Q)

with S(x) = 1/(1 + exp(-β(x - θ))). Depending on parameters the system
exhibits steady-state, limit-cycle oscillation, or chaos.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def sigmoid(x: float, beta: float = 1.0, theta: float = 0.0) -> float:
    return 1.0 / (1.0 + np.exp(-beta * (x - theta)))


@dataclass
class WilsonCowan:
    w_EE: float = 16.0
    w_EI: float = 12.0
    w_IE: float = 15.0
    w_II: float = 3.0
    tau_E: float = 10.0
    tau_I: float = 10.0
    beta: float = 1.3
    theta_E: float = 4.0
    theta_I: float = 3.7
    E: float = 0.1
    I: float = 0.1

    def step(self, P: float = 0.0, Q: float = 0.0,
              dt: float = 0.1) -> tuple[float, float]:
        S_E = sigmoid(self.w_EE * self.E - self.w_EI * self.I + P,
                      beta=self.beta, theta=self.theta_E)
        S_I = sigmoid(self.w_IE * self.E - self.w_II * self.I + Q,
                      beta=self.beta, theta=self.theta_I)
        dE = (-self.E + (1 - self.E) * S_E) / self.tau_E
        dI = (-self.I + (1 - self.I) * S_I) / self.tau_I
        self.E += dt * dE
        self.I += dt * dI
        return self.E, self.I

    def run(self, n_steps: int = 1000, P: float = 1.0, Q: float = 0.0,
             dt: float = 0.1) -> dict:
        Es = np.zeros(n_steps); Is = np.zeros(n_steps)
        for t in range(n_steps):
            e, i = self.step(P=P, Q=Q, dt=dt)
            Es[t] = e; Is[t] = i
        return {"E": Es, "I": Is}
