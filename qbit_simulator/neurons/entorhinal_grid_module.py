"""Entorhinal grid-cell module — multi-scale 2D periodic code.

In medial entorhinal cortex (mEC), neurons fire on a triangular grid.
Multiple modules exist at different spatial scales — finer scales give
precise localization, coarser scales give broad context.

This module: build a grid module = a set of grid cells whose firing
fields share the same scale λ and orientation θ but differ in phase.
Decoding by population-vector across phases gives a fine 2D position.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GridModule:
    n_cells: int = 16
    scale: float = 1.0
    orientation: float = 0.0
    phases: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.phases is None:
            # Phases uniformly within the rhombic unit cell.
            rng = np.random.default_rng(0)
            self.phases = rng.uniform(0, 1, size=(self.n_cells, 2))

    def _basis(self) -> np.ndarray:
        """Three k-vectors at 60° spacing (triangular lattice)."""
        th0 = self.orientation
        ks = []
        for k in range(3):
            ang = th0 + k * np.pi / 3
            ks.append((2 * np.pi / self.scale) * np.array([np.cos(ang), np.sin(ang)]))
        return np.array(ks)         # (3, 2)

    def firing(self, pos: np.ndarray) -> np.ndarray:
        """Per-cell firing rate at position pos = (x, y)."""
        ks = self._basis()
        rates = np.zeros(self.n_cells)
        for i in range(self.n_cells):
            phase_offset = self.phases[i] * self.scale
            r_eff = pos - phase_offset
            cos_sum = sum(np.cos(k @ r_eff) for k in ks)
            rates[i] = np.maximum(cos_sum + 1.5, 0.0) / 4.5    # normalize ≈ [0, 1]
        return rates


@dataclass
class GridSystem:
    """A stack of grid modules at different scales (geometric progression)."""
    scales: list = field(default_factory=lambda: [1.0, 1.5, 2.25, 3.375])
    n_cells_per_module: int = 16
    orientation: float = 0.0
    modules: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.modules:
            for s in self.scales:
                self.modules.append(GridModule(
                    n_cells=self.n_cells_per_module,
                    scale=s,
                    orientation=self.orientation))

    def firing(self, pos: np.ndarray) -> np.ndarray:
        return np.concatenate([m.firing(pos) for m in self.modules])

    def population_code(self, pos: np.ndarray) -> dict:
        rates = self.firing(pos)
        return {"rates": rates, "n_total": len(rates)}
