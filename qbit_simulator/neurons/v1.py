"""Primary visual cortex (V1) — Gabor filter bank.

Hubel & Wiesel discovered that V1 neurons respond to oriented edges.
The classic model: each neuron's receptive field is a 2D Gabor
function

    g(x, y; θ, σ, λ, ψ) = exp(-(x'^2 + γ^2 y'^2)/(2σ^2)) cos(2π x'/λ + ψ)

with x' = x cos θ + y sin θ, y' = -x sin θ + y cos θ.

Simple cells: linear Gabor response (with rectification).
Complex cells: pool simple cell responses across phase (sum of
squares of quadrature pair).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .retina import conv2d_separable


def gabor_filter(size: int, theta: float, sigma: float = 2.0,
                  lambd: float = 4.0, gamma: float = 0.5,
                  psi: float = 0.0) -> np.ndarray:
    half = size // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1]
    xp =  x * np.cos(theta) + y * np.sin(theta)
    yp = -x * np.sin(theta) + y * np.cos(theta)
    env = np.exp(-(xp * xp + gamma * gamma * yp * yp) / (2 * sigma * sigma))
    carrier = np.cos(2 * np.pi * xp / lambd + psi)
    return env * carrier


def convolve2d_full(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Naive 2D 'same' convolution via FFT."""
    H, W = img.shape
    kh, kw = kernel.shape
    pad_h = kh // 2; pad_w = kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    out = np.zeros_like(img, dtype=np.float64)
    # Direct conv (small kernels OK).
    for i in range(H):
        for j in range(W):
            patch = padded[i:i + kh, j:j + kw]
            out[i, j] = (patch * kernel).sum()
    return out


@dataclass
class V1Cortex:
    n_orientations: int = 8
    size: int = 11
    sigma: float = 2.0
    lambd: float = 4.0

    def filters(self) -> tuple[list, list]:
        """Quadrature pair (even and odd) at each orientation."""
        thetas = np.linspace(0, np.pi, self.n_orientations, endpoint=False)
        even = [gabor_filter(self.size, th, sigma=self.sigma,
                              lambd=self.lambd, psi=0.0)
                for th in thetas]
        odd  = [gabor_filter(self.size, th, sigma=self.sigma,
                              lambd=self.lambd, psi=np.pi / 2)
                for th in thetas]
        return even, odd

    def simple_responses(self, image: np.ndarray) -> np.ndarray:
        even, _ = self.filters()
        return np.stack([convolve2d_full(image, e) for e in even], axis=0)

    def complex_responses(self, image: np.ndarray) -> np.ndarray:
        even, odd = self.filters()
        out = np.zeros((self.n_orientations, *image.shape))
        for k in range(self.n_orientations):
            e = convolve2d_full(image, even[k])
            o = convolve2d_full(image, odd[k])
            out[k] = np.sqrt(e * e + o * o)
        return out

    def preferred_orientation(self, image: np.ndarray) -> np.ndarray:
        """Per-pixel argmax orientation (in radians)."""
        r = self.complex_responses(image)
        idx = r.argmax(axis=0)
        thetas = np.linspace(0, np.pi, self.n_orientations, endpoint=False)
        return thetas[idx]
