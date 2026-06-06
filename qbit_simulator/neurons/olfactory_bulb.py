"""Olfactory bulb — mitral / granule cell network.

Architecture:
  - N_M mitral cells receive sensory input from receptor cells.
  - N_G granule cells form a recurrent inhibitory network with mitral
    cells (granule cells lateral-inhibit mitrals via dendro-dendritic
    synapses).
  - The mitral-granule loop produces gamma oscillations (~40 Hz) and
    sharpens odor representations via lateral inhibition.

Output: sparse, decorrelated odor code in mitral cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class OlfactoryBulb:
    n_mitral: int = 50
    n_granule: int = 100
    tau_m: float = 5.0
    tau_g: float = 10.0
    W_mg: np.ndarray = field(default=None, repr=False)  # mitral -> granule
    W_gm: np.ndarray = field(default=None, repr=False)  # granule -> mitral (inhibitory)
    m: np.ndarray = field(default=None, repr=False)
    g: np.ndarray = field(default=None, repr=False)
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    def __post_init__(self) -> None:
        if self.W_mg is None:
            self.W_mg = self.rng.uniform(0, 0.1, (self.n_granule, self.n_mitral))
        if self.W_gm is None:
            self.W_gm = self.W_mg.T.copy()  # reciprocal
        if self.m is None:
            self.m = np.zeros(self.n_mitral)
        if self.g is None:
            self.g = np.zeros(self.n_granule)

    def step(self, sensory_input: np.ndarray, dt: float = 1.0) -> dict:
        dm = (-self.m + sensory_input - self.W_gm @ self.g) / self.tau_m
        dg = (-self.g + self.W_mg @ np.maximum(self.m, 0)) / self.tau_g
        self.m += dt * dm
        self.g += dt * dg
        return {"mitral": self.m.copy(), "granule": self.g.copy()}

    def run(self, odor: np.ndarray, n_steps: int = 200,
             dt: float = 1.0) -> dict:
        Ms = np.zeros((n_steps, self.n_mitral))
        Gs = np.zeros((n_steps, self.n_granule))
        for t in range(n_steps):
            out = self.step(odor, dt=dt)
            Ms[t] = out["mitral"]; Gs[t] = out["granule"]
        return {"mitral": Ms, "granule": Gs}
