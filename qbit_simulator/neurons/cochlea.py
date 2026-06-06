"""Cochlear / auditory nerve model.

The cochlea performs a real-time frequency analysis: hair cells along
the basilar membrane are tuned to different frequencies (low at the
apex, high at the base), and convert their local vibration into spike
trains in the auditory nerve.

Model:
  - Filter bank: a bank of bandpass filters (Gammatone-like) with
    center frequencies on the ERB scale.
  - Half-wave rectification + low-pass envelope.
  - LIF spike emission per filter channel.

Output: a "cochleagram" matrix (n_freq × n_steps), and optionally
spike trains.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def erb_scale(n: int, f_min: float = 50.0, f_max: float = 8000.0
              ) -> np.ndarray:
    """ERB-spaced center frequencies."""
    erb_min = 21.4 * np.log10(1 + 0.00437 * f_min)
    erb_max = 21.4 * np.log10(1 + 0.00437 * f_max)
    erbs = np.linspace(erb_min, erb_max, n)
    return (10 ** (erbs / 21.4) - 1) / 0.00437


def gammatone_impulse(fc: float, fs: float, length: float = 0.05,
                       order: int = 4) -> np.ndarray:
    """Gammatone impulse response: t^(N-1) exp(-2π b t) cos(2π fc t)."""
    n = int(length * fs)
    t = np.arange(n) / fs
    erb = 24.7 * (4.37e-3 * fc + 1)
    b = 1.019 * erb
    g = t ** (order - 1) * np.exp(-2 * np.pi * b * t) * np.cos(2 * np.pi * fc * t)
    return g / (np.abs(g).sum() + 1e-12)


@dataclass
class Cochlea:
    n_channels: int = 32
    f_min: float = 50.0
    f_max: float = 8000.0
    fs: float = 16000.0

    def __post_init__(self) -> None:
        self.center_freqs = erb_scale(self.n_channels, self.f_min, self.f_max)
        self.filters = [gammatone_impulse(fc, self.fs) for fc in self.center_freqs]

    def process(self, signal: np.ndarray) -> np.ndarray:
        """Returns cochleagram of shape (n_channels, n_samples)."""
        out = np.zeros((self.n_channels, len(signal)))
        for k, h in enumerate(self.filters):
            y = np.convolve(signal, h, mode="same")
            # Half-wave rectify + low-pass envelope (1 kHz cutoff).
            y = np.maximum(y, 0)
            # Exponential smoothing as low-pass.
            tau = self.fs / 1000.0
            alpha = np.exp(-1.0 / tau)
            env = np.zeros_like(y)
            for i, v in enumerate(y):
                env[i] = alpha * (env[i - 1] if i > 0 else 0) + (1 - alpha) * v
            out[k] = env
        return out

    def encode_spikes(self, signal: np.ndarray, n_steps: int = 200,
                       rng: np.random.Generator | None = None) -> np.ndarray:
        """Bin cochleagram into n_steps frames, Poisson-encode rates."""
        rng = rng or np.random.default_rng(0)
        cgram = self.process(signal)
        # Bin into n_steps.
        L = cgram.shape[1]
        bin_size = max(L // n_steps, 1)
        binned = np.zeros((self.n_channels, n_steps))
        for t in range(n_steps):
            lo = t * bin_size
            hi = min(lo + bin_size, L)
            binned[:, t] = cgram[:, lo:hi].mean(axis=1)
        # Normalize to [0, 1] per channel as Poisson rate.
        m = binned.max() + 1e-12
        binned /= m
        spikes = rng.uniform(size=binned.shape) < binned
        return spikes


def make_tone(freq: float, duration: float = 0.5,
              fs: float = 16000.0, amp: float = 1.0) -> np.ndarray:
    """Pure-tone test signal."""
    t = np.arange(int(duration * fs)) / fs
    return amp * np.sin(2 * np.pi * freq * t)
