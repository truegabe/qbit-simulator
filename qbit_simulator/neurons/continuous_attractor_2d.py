"""2D continuous attractor network ("CAN" / bump attractor).

Models like the rodent head-direction system and grid cell modules
implement bump attractors on a 2D sheet: locally excitatory, globally
inhibitory connectivity supports a stable activity "bump" that can
be moved smoothly by external drive.

Connectivity:
    W(i, j) = J_exc · exp(-d^2/(2 sigma_E^2)) - J_inh
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ContinuousAttractor2D:
    h: int = 20
    w: int = 20
    J_exc: float = 2.0
    J_inh: float = 0.3
    sigma_E: float = 2.0
    tau: float = 10.0
    rate: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.rate is None:
            self.rate = np.zeros((self.h, self.w))

    def _distance_matrix(self) -> np.ndarray:
        # Periodic boundary distance on h × w torus.
        rows = np.arange(self.h)
        cols = np.arange(self.w)
        dr = np.minimum(np.abs(rows[:, None] - rows[None, :]),
                         self.h - np.abs(rows[:, None] - rows[None, :]))
        dc = np.minimum(np.abs(cols[:, None] - cols[None, :]),
                         self.w - np.abs(cols[:, None] - cols[None, :]))
        return dr, dc

    def kernel(self) -> np.ndarray:
        """Build the convolution kernel on a centered grid."""
        rows = (np.arange(self.h) - self.h // 2)
        cols = (np.arange(self.w) - self.w // 2)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        d2 = rr * rr + cc * cc
        return self.J_exc * np.exp(-d2 / (2 * self.sigma_E ** 2)) - self.J_inh

    def step(self, external: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """One step of network dynamics.

        Uses FFT-based circular convolution for the recurrent input.
        """
        K = self.kernel()
        # Circular conv via FFT (zero-mean shifted kernel).
        K_shift = np.roll(K, shift=(-self.h // 2, -self.w // 2), axis=(0, 1))
        recurrent = np.real(np.fft.ifft2(np.fft.fft2(self.rate) * np.fft.fft2(K_shift)))
        drive = recurrent + external
        # Linear-threshold dynamics.
        target = np.maximum(drive, 0.0)
        self.rate += dt * (target - self.rate) / self.tau
        return self.rate

    def bump_center(self) -> tuple[float, float]:
        """Population vector decode of bump center (periodic)."""
        rows = np.arange(self.h)
        cols = np.arange(self.w)
        # Use circular mean: angle = atan2(sum sin, sum cos).
        sin_r = np.sin(2 * np.pi * rows / self.h)
        cos_r = np.cos(2 * np.pi * rows / self.h)
        sin_c = np.sin(2 * np.pi * cols / self.w)
        cos_c = np.cos(2 * np.pi * cols / self.w)
        Z = self.rate.sum() + 1e-12
        row_marg = self.rate.sum(axis=1)
        col_marg = self.rate.sum(axis=0)
        ang_r = np.arctan2((row_marg * sin_r).sum() / Z,
                            (row_marg * cos_r).sum() / Z)
        ang_c = np.arctan2((col_marg * sin_c).sum() / Z,
                            (col_marg * cos_c).sum() / Z)
        r = (ang_r / (2 * np.pi)) * self.h % self.h
        c = (ang_c / (2 * np.pi)) * self.w % self.w
        return float(r), float(c)
