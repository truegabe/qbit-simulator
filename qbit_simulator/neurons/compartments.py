"""Multi-compartment neurons (cable equation).

Real neurons have spatially-extended dendrites where signals attenuate
as they propagate. A simple model: discretize the dendritic tree into
N compartments, each with its own V_i, connected by axial conductances
g_axial.

For each compartment i:
    C_i dV_i/dt = -g_L (V_i - E_L)
                  + sum_j g_axial(i, j) (V_j - V_i)
                  + I_syn(i) + I_inj(i)

The soma (compartment 0) has spike-generating sodium dynamics
(simplified as an LIF-style threshold).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Compartment:
    """One dendritic compartment."""
    C: float = 1.0
    g_L: float = 0.05
    E_L: float = -70.0
    V: float = -70.0


@dataclass
class MultiCompartmentNeuron:
    """A multi-compartment neuron.

    Topology: a list of (i, j, g_axial) couplings. Compartment 0 is
    treated as the soma (spike emission). All others are dendritic.
    """
    n: int
    C:  np.ndarray = field(default=None, repr=False)
    g_L: np.ndarray = field(default=None, repr=False)
    E_L: float = -70.0
    V:  np.ndarray = field(default=None, repr=False)
    g_axial: np.ndarray = field(default=None, repr=False)   # (n, n)

    V_th: float = -50.0
    V_reset: float = -70.0
    t_refrac: int = 2
    refrac_until: int = -1

    def __post_init__(self) -> None:
        if self.C is None:
            self.C = np.ones(self.n)
        if self.g_L is None:
            self.g_L = np.full(self.n, 0.05)
        if self.V is None:
            self.V = np.full(self.n, self.E_L)
        if self.g_axial is None:
            # Default: linear chain 0-1-2-...-(n-1).
            G = np.zeros((self.n, self.n))
            for i in range(self.n - 1):
                G[i, i + 1] = G[i + 1, i] = 0.5
            self.g_axial = G
        else:
            # Symmetrize.
            self.g_axial = 0.5 * (self.g_axial + self.g_axial.T)

    def step(self, I_inj: np.ndarray, t: int, dt: float = 1.0) -> bool:
        """One Euler step. Returns True on soma spike."""
        # Axial currents: I_axial(i) = sum_j g(i,j) (V_j - V_i).
        # Vectorized: g_axial @ V  - (sum_j g_axial(i,j)) * V_i.
        row_sum = self.g_axial.sum(axis=1)
        I_axial = self.g_axial @ self.V - row_sum * self.V
        I_leak = -self.g_L * (self.V - self.E_L)
        # Soma in refractory?
        if t <= self.refrac_until:
            self.V[0] = self.V_reset
            # Dendrites keep evolving.
            dV = dt * (I_leak[1:] + I_axial[1:] + I_inj[1:]) / self.C[1:]
            self.V[1:] += dV
            return False
        dV = dt * (I_leak + I_axial + I_inj) / self.C
        self.V += dV
        if self.V[0] >= self.V_th:
            self.V[0] = self.V_reset
            self.refrac_until = t + self.t_refrac
            return True
        return False


def linear_dendrite(n: int, g_axial: float = 0.5,
                    soma_C: float = 1.0,
                    dend_C: float = 0.5) -> MultiCompartmentNeuron:
    """A neuron with linear dendritic chain. Compartment 0 = soma."""
    C = np.full(n, dend_C); C[0] = soma_C
    G = np.zeros((n, n))
    for i in range(n - 1):
        G[i, i + 1] = G[i + 1, i] = g_axial
    return MultiCompartmentNeuron(n=n, C=C, g_axial=G)


def attenuation(neuron: MultiCompartmentNeuron, dt: float = 1.0,
                 n_steps: int = 200, drive_compartment: int = -1,
                 amplitude: float = 5.0) -> np.ndarray:
    """Drive a distal compartment, measure steady-state attenuation along
    the cable. Returns V at the end of the sim per compartment."""
    if drive_compartment < 0:
        drive_compartment = neuron.n - 1
    I = np.zeros(neuron.n)
    I[drive_compartment] = amplitude
    for t in range(n_steps):
        neuron.step(I, t=t, dt=dt)
    return neuron.V.copy()
