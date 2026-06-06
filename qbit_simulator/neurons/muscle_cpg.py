"""Hill muscle model + spinal central pattern generator (CPG).

Hill-type muscle (active force = a · f_L(L) · f_V(V)):
  - a: activation (∈ [0,1])
  - f_L: force-length relationship (Gaussian-like)
  - f_V: force-velocity relationship (Hill hyperbola, simplified)

CPG: a pair of mutually-inhibiting half-centers (e.g. flexor /
extensor) produces alternating rhythmic output without any oscillatory
input. Implemented here with two leaky-integrator neurons + reciprocal
inhibition + spike-frequency adaptation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def hill_force(activation: float, length: float, velocity: float,
                L_opt: float = 1.0, width: float = 0.4,
                V_max: float = 10.0) -> float:
    """Total muscle force = a · f_L · f_V."""
    f_L = np.exp(-((length - L_opt) / width) ** 2)
    # Force-velocity: hyperbolic.
    if velocity >= 0:
        f_V = (V_max - velocity) / (V_max + velocity * 3.0)
    else:
        # Eccentric, force can exceed isometric.
        f_V = 1.5 - 0.5 * (V_max + velocity) / (V_max - velocity * 3.0)
    return float(max(activation * f_L * f_V, 0.0))


@dataclass
class HalfCenterCPG:
    """Two mutually-inhibitory neurons with adaptation."""
    tau: float = 50.0
    tau_a: float = 200.0
    w_inh: float = 1.0
    drive: float = 1.0
    threshold: float = 0.5
    x: np.ndarray = field(default_factory=lambda: np.array([0.6, 0.4]))
    a: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def step(self, dt: float = 1.0) -> np.ndarray:
        # Sigmoid output.
        y = 1.0 / (1.0 + np.exp(-10 * (self.x - self.threshold)))
        # Reciprocal inhibition: x_0 receives -w * y_1, x_1 receives -w * y_0.
        inh = np.array([y[1], y[0]]) * self.w_inh
        dx = (-self.x + self.drive - inh - self.a) / self.tau
        da = (-self.a + 1.5 * y) / self.tau_a
        self.x += dt * dx
        self.a += dt * da
        return y

    def run(self, n_steps: int = 1000, dt: float = 1.0) -> np.ndarray:
        out = np.zeros((n_steps, 2))
        for t in range(n_steps):
            out[t] = self.step(dt=dt)
        return out


@dataclass
class MuscleArmModel:
    """Single 1-DOF arm: theta is joint angle, two antagonistic muscles."""
    inertia: float = 1.0
    damping: float = 0.5
    theta: float = 0.0
    theta_dot: float = 0.0

    def step(self, a_flex: float, a_ext: float, dt: float = 0.01) -> float:
        # Each muscle's length depends on angle; one shortens as the
        # other lengthens.
        L_flex = 1.0 - 0.3 * np.sin(self.theta)
        L_ext  = 1.0 + 0.3 * np.sin(self.theta)
        V_flex = -0.3 * np.cos(self.theta) * self.theta_dot
        V_ext  =  0.3 * np.cos(self.theta) * self.theta_dot
        F_flex = hill_force(a_flex, L_flex, V_flex)
        F_ext  = hill_force(a_ext,  L_ext,  V_ext)
        torque = 0.3 * (F_flex - F_ext) - self.damping * self.theta_dot
        self.theta_dot += dt * torque / self.inertia
        self.theta += dt * self.theta_dot
        return self.theta
