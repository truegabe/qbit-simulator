"""Retina model.

Simplified retinal processing:
  - Photoreceptor: image → log luminance.
  - Horizontal cells: local averaging (low-pass).
  - Bipolar cells: ON/OFF center-surround via Difference-of-Gaussians
    (DoG) — the classic retinal ganglion cell receptive field.
  - Ganglion cells: spike via LIF on the bipolar output.

Output: spike-rate "image" or actual LIF spike trains.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    if size % 2 == 0:
        size += 1
    r = np.arange(size) - size // 2
    g = np.exp(-r * r / (2 * sigma * sigma))
    return g / g.sum()


def conv2d_separable(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur via separable convolution with reflect padding."""
    k = gaussian_kernel(int(6 * sigma + 1), sigma)
    pad = len(k) // 2
    padded = np.pad(img, pad, mode="reflect")
    # Horizontal then vertical.
    tmp = np.zeros_like(img, dtype=np.float64)
    for i in range(img.shape[0]):
        tmp[i] = np.convolve(padded[i + pad], k, mode="valid")
    out = np.zeros_like(img, dtype=np.float64)
    padded2 = np.pad(tmp, pad, mode="reflect")
    for j in range(img.shape[1]):
        out[:, j] = np.convolve(padded2[:, j + pad], k, mode="valid")
    return out


@dataclass
class Retina:
    sigma_center: float = 1.0
    sigma_surround: float = 3.0
    threshold: float = 0.0
    on_off: bool = True   # if True, return (on, off) channels separately

    def photoreceptor(self, image: np.ndarray) -> np.ndarray:
        """Log compression."""
        return np.log1p(image.astype(np.float64))

    def dog(self, img: np.ndarray) -> np.ndarray:
        """Difference-of-Gaussians."""
        center = conv2d_separable(img, self.sigma_center)
        surround = conv2d_separable(img, self.sigma_surround)
        return center - surround

    def __call__(self, image: np.ndarray) -> dict:
        x = self.photoreceptor(image)
        d = self.dog(x)
        if self.on_off:
            on  = np.maximum(d - self.threshold, 0.0)
            off = np.maximum(-d - self.threshold, 0.0)
            return {"on": on, "off": off, "dog": d}
        return {"rate": np.maximum(d - self.threshold, 0.0), "dog": d}


def encode_to_spikes(rates: np.ndarray, n_steps: int,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """Poisson spike encoding. Rates should be in [0, 1] per step."""
    rng = rng or np.random.default_rng(0)
    spikes = rng.uniform(size=(n_steps, *rates.shape)) < rates
    return spikes
