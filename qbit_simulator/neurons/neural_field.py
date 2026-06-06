"""Amari neural field equation (continuum cortex).

    τ ∂u(x, t)/∂t = -u(x, t) + ∫ w(x - x') f(u(x', t)) dx' + I(x, t)

with f a sigmoid firing-rate function and w a Mexican-hat connectivity
(positive at short range, negative at long range). The equation
supports localized bumps, traveling waves, and breathers — model of
visual hallucinations, working memory bumps, and cortical pattern
formation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def mexican_hat(x: np.ndarray, A_e: float = 1.0, A_i: float = 0.5,
                 sigma_e: float = 1.0, sigma_i: float = 3.0) -> np.ndarray:
    return (A_e * np.exp(-x * x / (2 * sigma_e ** 2))
            - A_i * np.exp(-x * x / (2 * sigma_i ** 2)))


def sigmoid(x: np.ndarray, beta: float = 1.0, theta: float = 0.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-beta * (x - theta)))


@dataclass
class NeuralField1D:
    L: int = 100
    dx: float = 0.5
    tau: float = 10.0
    beta: float = 5.0
    theta: float = 0.5
    u: np.ndarray = field(default=None, repr=False)
    kernel: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.u is None:
            self.u = np.zeros(self.L)
        if self.kernel is None:
            x = (np.arange(self.L) - self.L // 2) * self.dx
            self.kernel = mexican_hat(x)

    def step(self, I: np.ndarray, dt: float = 1.0) -> np.ndarray:
        f_u = sigmoid(self.u, beta=self.beta, theta=self.theta)
        # FFT-based circular conv with centered kernel.
        K_shift = np.roll(self.kernel, -self.L // 2)
        conv = np.real(np.fft.ifft(np.fft.fft(f_u) * np.fft.fft(K_shift))) * self.dx
        du = (-self.u + conv + I) / self.tau
        self.u += dt * du
        return self.u

    def run(self, I_func, n_steps: int = 500, dt: float = 1.0) -> np.ndarray:
        out = np.zeros((n_steps, self.L))
        for t in range(n_steps):
            I = I_func(t) if callable(I_func) else I_func
            out[t] = self.step(I, dt=dt).copy()
        return out
