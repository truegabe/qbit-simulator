"""Single qubit as 4 real neurons: [alpha_re, alpha_im, beta_re, beta_im]."""

from __future__ import annotations

import numpy as np


class Qubit:
    __slots__ = ("neurons",)

    def __init__(self, alpha: complex = 1 + 0j, beta: complex = 0 + 0j):
        self.neurons = np.array(
            [alpha.real, alpha.imag, beta.real, beta.imag], dtype=np.float64
        )
        self.normalize()

    @classmethod
    def zero(cls) -> "Qubit":
        return cls(1 + 0j, 0 + 0j)

    @classmethod
    def one(cls) -> "Qubit":
        return cls(0 + 0j, 1 + 0j)

    @classmethod
    def plus(cls) -> "Qubit":
        s = 1 / np.sqrt(2)
        return cls(s + 0j, s + 0j)

    @classmethod
    def minus(cls) -> "Qubit":
        s = 1 / np.sqrt(2)
        return cls(s + 0j, -s + 0j)

    @property
    def alpha(self) -> complex:
        return complex(self.neurons[0], self.neurons[1])

    @property
    def beta(self) -> complex:
        return complex(self.neurons[2], self.neurons[3])

    @property
    def state(self) -> np.ndarray:
        return np.array([self.alpha, self.beta], dtype=np.complex128)

    def prob_zero(self) -> float:
        return float(self.neurons[0] ** 2 + self.neurons[1] ** 2)

    def prob_one(self) -> float:
        return float(self.neurons[2] ** 2 + self.neurons[3] ** 2)

    def normalize(self) -> "Qubit":
        norm = np.linalg.norm(self.neurons)
        if norm < 1e-15:
            raise ValueError("Cannot normalize zero state.")
        self.neurons /= norm
        return self

    def apply(self, gate_2x2: np.ndarray, renormalize: bool = False) -> "Qubit":
        new_state = gate_2x2 @ self.state
        self.neurons[0] = new_state[0].real
        self.neurons[1] = new_state[0].imag
        self.neurons[2] = new_state[1].real
        self.neurons[3] = new_state[1].imag
        if renormalize:
            self.normalize()
        return self

    def __repr__(self) -> str:
        return f"Qubit({self.alpha:.4g}|0> + {self.beta:.4g}|1>)"
