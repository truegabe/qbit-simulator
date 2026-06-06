"""Place cells and grid cells: spatial-navigation primitives.

The hippocampus and entorhinal cortex contain neurons that fire at
specific spatial locations (PLACE CELLS, O'Keefe 1971) or on a
hexagonal lattice (GRID CELLS, Hafting et al. 2005). Together they
implement a "cognitive map" for spatial navigation.

This module provides:

  - `PlaceCell(center, sigma)`: Gaussian place field — fires maximally
    at one location.
  - `PlaceCellPopulation(n_cells, env_size)`: a population tiling a 2D
    environment with overlapping place fields.
  - `GridCell(scale, orientation, offset)`: hexagonal grid-cell firing
    rate.
  - `GridCellPopulation(n_modules, n_cells_per_module)`: stacked grid
    modules at different scales — provides a multi-scale spatial code.
  - `ContinuousAttractor(n_cells)`: a ring/grid attractor that
    INTEGRATES velocity input into a position bump (path integration).
  - `decode_position(activity, cells)`: population-vector decoder.

Spatial positions are 2D (x, y) ∈ [0, env_size]².
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------------
# Place cells
# ----------------------------------------------------------------------------

@dataclass
class PlaceCell:
    """A single place cell with a Gaussian receptive field."""
    center: np.ndarray              # (x, y)
    sigma:  float = 0.5             # field width
    peak_rate: float = 1.0          # max firing rate

    def firing_rate(self, position: np.ndarray) -> float:
        """f(pos) = peak · exp(-||pos − center||² / (2·sigma²))."""
        d = position - self.center
        r2 = float(d @ d)
        return float(self.peak_rate * np.exp(-r2 / (2 * self.sigma ** 2)))


@dataclass
class PlaceCellPopulation:
    """N place cells tiling a 2D environment.

    By default, cells are arranged on a regular grid in the environment.
    """
    n_cells:  int = 25         # √n_cells × √n_cells grid
    env_size: float = 5.0
    sigma:    float = 0.8
    cells:    list[PlaceCell] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.cells:
            side = int(np.ceil(np.sqrt(self.n_cells)))
            xs = np.linspace(0, self.env_size, side)
            ys = np.linspace(0, self.env_size, side)
            for x in xs:
                for y in ys:
                    if len(self.cells) < self.n_cells:
                        self.cells.append(
                            PlaceCell(center=np.array([x, y]),
                                       sigma=self.sigma)
                        )

    def activity(self, position: np.ndarray) -> np.ndarray:
        """Per-cell firing rates at the given position."""
        return np.array([c.firing_rate(position) for c in self.cells])

    def decode_position(self, activity: np.ndarray) -> np.ndarray:
        """Population-vector decoder: weighted average of cell centers."""
        if activity.sum() < 1e-9:
            return np.zeros(2)
        weights = activity / activity.sum()
        positions = np.array([c.center for c in self.cells])
        return weights @ positions


# ----------------------------------------------------------------------------
# Grid cells
# ----------------------------------------------------------------------------

@dataclass
class GridCell:
    """A single grid cell with a hexagonal-periodic firing pattern.

    Standard model: rate is the sum of three plane waves at 60° angles,
    scaled by lattice spacing `scale` and rotated by `orientation`,
    offset by `offset`.
    """
    scale: float = 1.0
    orientation: float = 0.0    # radians
    offset: np.ndarray = field(default_factory=lambda: np.zeros(2))
    peak_rate: float = 1.0

    def firing_rate(self, position: np.ndarray) -> float:
        x, y = position - self.offset
        # Three plane-wave directions 60° apart.
        directions = [self.orientation + k * np.pi / 3 for k in range(3)]
        k_mag = 2 * np.pi / self.scale
        r = 0.0
        for theta in directions:
            kx, ky = k_mag * np.cos(theta), k_mag * np.sin(theta)
            r += np.cos(kx * x + ky * y)
        # Shift to [0, 1] range and rectify.
        r = max(0.0, (r + 1.5) / 4.5)
        return float(self.peak_rate * r)


@dataclass
class GridCellPopulation:
    """Multi-module grid-cell population.

    Each module has cells at the same scale + orientation, but offset
    randomly. Different modules have different scales (typically scaled
    by √2 in real entorhinal cortex).
    """
    n_modules: int = 3
    n_cells_per_module: int = 12
    base_scale: float = 1.0
    scale_ratio: float = np.sqrt(2)
    rng_seed: int = 0
    cells: list[GridCell] = field(default_factory=list)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.rng_seed)
        if not self.cells:
            for m in range(self.n_modules):
                scale = self.base_scale * (self.scale_ratio ** m)
                for _ in range(self.n_cells_per_module):
                    self.cells.append(GridCell(
                        scale=scale,
                        orientation=rng.uniform(0, np.pi / 3),
                        offset=rng.uniform(0, scale, size=2),
                    ))

    def activity(self, position: np.ndarray) -> np.ndarray:
        return np.array([c.firing_rate(position) for c in self.cells])


# ----------------------------------------------------------------------------
# Continuous attractor for path integration
# ----------------------------------------------------------------------------

@dataclass
class RingAttractor:
    """1D ring attractor: n neurons arranged on a ring, recurrent
    connectivity creates a stable "bump" of activity that can be moved
    by velocity input.

    Equation:  τ · dr_i/dt = -r_i + f(∑_j W_ij r_j + I_i)
    with W having a "Mexican-hat" shape (close excitation, far inhibition).
    """
    n: int = 30
    tau: float = 10.0
    w_excite: float = 1.5
    w_inhibit: float = -0.5
    excite_sigma: float = 3.0    # in units of neurons

    state: np.ndarray = field(default=None, repr=False)
    weights: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.state is None:
            self.state = np.zeros(self.n)
        if self.weights is None:
            W = np.zeros((self.n, self.n))
            for i in range(self.n):
                for j in range(self.n):
                    d = min(abs(i - j), self.n - abs(i - j))  # ring distance
                    W[i, j] = (self.w_excite * np.exp(-d ** 2 / (2 * self.excite_sigma ** 2))
                                + self.w_inhibit / self.n)
            self.weights = W

    def reset(self) -> None:
        self.state[:] = 0

    def initialize_bump(self, center: int, width: float = 3.0) -> None:
        """Seed a Gaussian bump centered at `center`."""
        idx = np.arange(self.n)
        d = np.minimum(np.abs(idx - center), self.n - np.abs(idx - center))
        self.state = np.exp(-d ** 2 / (2 * width ** 2))

    def step(self, external_input: np.ndarray | None = None,
              velocity: float = 0.0, dt: float = 1.0) -> np.ndarray:
        """One Euler step.

        Args:
            external_input: per-neuron extra input.
            velocity:       direct shift of the bump by `velocity·dt`
                            neurons per step (path integration).
            dt:             time step.

        Returns:
            new state (firing rates).
        """
        I = self.weights @ self.state
        if external_input is not None:
            I = I + external_input
        # Mexican-hat dynamics.
        self.state = self.state + (dt / self.tau) * (-self.state + np.tanh(I))
        # Velocity shift: directly translate the bump on the ring.
        # In real entorhinal cortex this is done via head-direction
        # neurons asymmetrically gating recurrent connections; the
        # net effect is the same translation.
        if abs(velocity) > 1e-9:
            shift = int(round(velocity * dt))
            if shift != 0:
                self.state = np.roll(self.state, shift)
        return self.state.copy()

    def estimate_bump_center(self) -> float:
        """Population-vector estimate of the bump position (in [0, n))."""
        if self.state.sum() < 1e-9:
            return 0.0
        angles = np.linspace(0, 2 * np.pi, self.n, endpoint=False)
        x = np.sum(self.state * np.cos(angles))
        y = np.sum(self.state * np.sin(angles))
        theta = np.arctan2(y, x) % (2 * np.pi)
        return float(theta * self.n / (2 * np.pi))


def integrate_path(attractor: RingAttractor,
                    velocity_sequence: list[float],
                    n_steps_per_velocity: int = 20) -> list[float]:
    """Drive a ring attractor with a sequence of velocities; record the
    bump center over time."""
    centers = [attractor.estimate_bump_center()]
    for v in velocity_sequence:
        for _ in range(n_steps_per_velocity):
            attractor.step(velocity=v)
        centers.append(attractor.estimate_bump_center())
    return centers
