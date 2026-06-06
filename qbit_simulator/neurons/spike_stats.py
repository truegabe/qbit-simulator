"""Spike-train statistics: ISI distribution, CV, Fano factor, PSTH.

The most common neural-data analyses for single neurons and
populations.

ISI distribution:
  Inter-spike intervals. For a Poisson process, ISIs are exponentially
  distributed.

Coefficient of variation:
  CV = std(ISI) / mean(ISI). Poisson → CV = 1. Regular firing → CV = 0.

Fano factor:
  F = Var(N) / Mean(N) where N is spike count in a window. Poisson → 1.

PSTH:
  Peri-stimulus time histogram. Spike rate as a function of time
  averaged across trials.
"""

from __future__ import annotations

import numpy as np


def isi(spike_train: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """Inter-spike intervals from a boolean (T,) train."""
    times = np.where(spike_train)[0]
    if len(times) < 2:
        return np.array([])
    return np.diff(times) * dt


def cv_isi(spike_train: np.ndarray) -> float:
    intervals = isi(spike_train)
    if len(intervals) < 2:
        return 0.0
    return float(intervals.std() / (intervals.mean() + 1e-12))


def fano_factor(spike_train: np.ndarray, window: int = 50) -> float:
    """Compute Fano factor over non-overlapping windows."""
    n_full = (spike_train.shape[0] // window) * window
    if n_full == 0:
        return 0.0
    counts = spike_train[:n_full].reshape(-1, window).sum(axis=1)
    m = counts.mean()
    if m < 1e-12:
        return 0.0
    return float(counts.var() / m)


def isi_histogram(spike_trains: list, bin_width: float = 1.0,
                   max_isi: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Pool ISIs from many trials and return histogram counts + bin edges."""
    all_isi = []
    for tr in spike_trains:
        all_isi.append(isi(tr))
    if not all_isi:
        return np.array([]), np.array([])
    pooled = np.concatenate(all_isi)
    if len(pooled) == 0:
        return np.array([]), np.array([])
    if max_isi is None:
        max_isi = float(pooled.max()) + bin_width
    bins = np.arange(0.0, max_isi + bin_width, bin_width)
    counts, edges = np.histogram(pooled, bins=bins)
    return counts, edges


def psth(spike_trains: np.ndarray, bin_width: int = 5,
          dt: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Peri-stimulus time histogram.

    spike_trains: shape (n_trials, T). Returns (rate per bin, bin centers).
    """
    n_trials, T = spike_trains.shape
    n_bins = T // bin_width
    counts = spike_trains[:, :n_bins * bin_width].reshape(
        n_trials, n_bins, bin_width).sum(axis=2)
    rate = counts.mean(axis=0) / (bin_width * dt)
    centers = (np.arange(n_bins) + 0.5) * bin_width * dt
    return rate, centers
