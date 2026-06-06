"""Axonal delay line -- conduction-velocity temporal coding.

Different axons have different diameters and myelination levels.
Conduction velocity scales with diameter:
  thin, unmyelinated (C-fibres):  ~0.5-2 m/s
  thick, myelinated (A-alpha):    ~70-120 m/s

This creates INTENTIONAL delays the brain exploits for:

  1. COINCIDENCE DETECTION (sound localisation in MSO/LSO):
     Signals from left and right ears travel different path lengths.
     The neuron that receives both simultaneously fires -- that neuron
     encodes the inter-aural time difference (ITD).

  2. TEMPORAL PATTERN MATCHING (cerebellum):
     Granule cell axons (parallel fibres) tap Purkinje cell dendrites
     at different delays.  The Purkinje cell fires maximally when the
     input pattern matches the delay-line template.

  3. PHASE COMPENSATION:
     Signals from distant cortical regions arrive later.
     Delay lines equalize travel time so regions stay in sync.

Architecture:
  Each signal dimension has a FIFO buffer of length = max_delay.
  Input x_t is pushed at time t; it emerges at time t + delay_i.
  Different dimensions can have different delays (heterogeneous).

Classes
-------
  DelayLine          -- single FIFO for one signal channel
  DelayLineBank      -- n_dims delay lines with per-channel delays
  AxonalDelayLine    -- full system: push activations, pop delayed outputs
  TemporalMatcher    -- matched filter: detects a stored spike-train template
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# DelayLine  (single channel FIFO)
# ---------------------------------------------------------------------------

class DelayLine:
    """Single-channel delay line.

    Buffers values for `delay` time steps.  Each call to `push(v)`
    adds v to the back; `pop()` removes and returns the front.

    If the buffer is not yet full (t < delay), pop() returns fill_value.
    """

    def __init__(self, delay: int, fill_value: float = 0.0) -> None:
        if delay < 0:
            raise ValueError("delay must be >= 0")
        self.delay      = delay
        self.fill_value = fill_value
        self._buf: deque = deque([fill_value] * delay, maxlen=delay if delay > 0 else 1)

    def push(self, value: float) -> float:
        """Push a new value and return the delayed output."""
        if self.delay == 0:
            return float(value)
        oldest = self._buf[0]
        self._buf.append(float(value))
        return oldest

    def peek(self) -> float:
        """Return next output without advancing."""
        return self._buf[0]

    def reset(self, fill_value: float = None) -> None:
        fv = fill_value if fill_value is not None else self.fill_value
        for i in range(len(self._buf)):
            self._buf[i] = fv


# ---------------------------------------------------------------------------
# DelayLineBank  (n parallel channels, each with its own delay)
# ---------------------------------------------------------------------------

@dataclass
class DelayLineBank:
    """Bank of n_dims parallel delay lines with heterogeneous delays.

    Parameters
    ----------
    delays : array-like of int, length n_dims
             Each value is the delay (in time steps) for that channel.
    """
    delays: np.ndarray = field(repr=False)

    _lines: list = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.delays = np.asarray(self.delays, dtype=int)
        self._lines = [DelayLine(int(d)) for d in self.delays]

    @property
    def n_dims(self) -> int:
        return len(self._lines)

    def push(self, x: np.ndarray) -> np.ndarray:
        """Push one time-step of activations, return delayed outputs."""
        x = np.asarray(x, dtype=np.float64).ravel()
        return np.array([line.push(v) for line, v in zip(self._lines, x)])

    def reset(self) -> None:
        for line in self._lines:
            line.reset()


# ---------------------------------------------------------------------------
# AxonalDelayLine  (full system with routing and stats)
# ---------------------------------------------------------------------------

@dataclass
class AxonalDelayLine:
    """Full axonal delay line system for a brain-region interface.

    Parameters
    ----------
    n_dims        : number of signal dimensions (neurons / channels)
    delay_mode    : how to assign delays per channel:
                    'uniform'   -- all channels get the same delay
                    'linear'    -- delays spread linearly from min to max
                    'random'    -- delays drawn from U[min_delay, max_delay]
                    'custom'    -- use delays_custom array
    min_delay     : minimum delay in time steps
    max_delay     : maximum delay in time steps
    delays_custom : explicit per-channel delays (used if mode='custom')
    """
    n_dims:        int
    delay_mode:    str   = "linear"
    min_delay:     int   = 1
    max_delay:     int   = 10
    delays_custom: np.ndarray = field(default=None, repr=False)
    rng: np.random.Generator  = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    bank: DelayLineBank = field(default=None, repr=False)
    _t:   int           = field(default=0, repr=False)

    def __post_init__(self) -> None:
        delays = self._make_delays()
        self.bank = DelayLineBank(delays=delays)

    def _make_delays(self) -> np.ndarray:
        n = self.n_dims
        if self.delay_mode == "uniform":
            return np.full(n, self.min_delay, dtype=int)
        if self.delay_mode == "linear":
            return np.linspace(self.min_delay, self.max_delay, n,
                               dtype=int)
        if self.delay_mode == "random":
            return self.rng.integers(self.min_delay,
                                     self.max_delay + 1, size=n)
        if self.delay_mode == "custom":
            if self.delays_custom is None:
                raise ValueError("delays_custom must be provided for mode='custom'")
            return np.asarray(self.delays_custom, dtype=int)
        raise ValueError(f"unknown delay_mode: {self.delay_mode}")

    @property
    def delays(self) -> np.ndarray:
        return self.bank.delays

    def push(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        """Push activation x at current time step, get delayed outputs.

        Returns
        -------
        delayed_x : np.ndarray (n_dims,) -- delayed signal
        info      : dict with time, mean_delay, delay_spread
        """
        x_del = self.bank.push(x)
        self._t += 1
        info = {
            "t":            self._t,
            "mean_delay":   float(self.delays.mean()),
            "delay_spread": float(self.delays.std()),
            "min_delay":    int(self.delays.min()),
            "max_delay":    int(self.delays.max()),
        }
        return x_del, info

    def reset(self) -> None:
        self.bank.reset()
        self._t = 0

    def delay_profile(self) -> np.ndarray:
        """Return per-channel delay values."""
        return self.bank.delays.copy()


# ---------------------------------------------------------------------------
# TemporalMatcher  (matched filter using delay lines)
# ---------------------------------------------------------------------------

@dataclass
class TemporalMatcher:
    """Matched filter for temporal spike-train patterns.

    Stores a template pattern (sequence of activations over time).
    Uses a sliding delay-line bank to compute the cross-correlation
    between the incoming stream and the template at every time step.

    When correlation peaks, the template pattern has been detected.

    Parameters
    ----------
    template      : 2D array (T_template, n_dims) -- the pattern to detect
    threshold     : detection threshold on normalised correlation
    """
    template:  np.ndarray
    threshold: float = 0.8
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    _T:    int           = field(init=False, repr=False)
    _bank: DelayLineBank = field(init=False, repr=False)
    _buf:  np.ndarray    = field(init=False, repr=False)  # sliding window

    def __post_init__(self) -> None:
        self.template = np.asarray(self.template, dtype=np.float64)
        if self.template.ndim == 1:
            self.template = self.template[:, None]
        self._T, n_dims = self.template.shape
        # A bank where channel i has delay i (tap at each past time step).
        self._bank = DelayLineBank(delays=np.arange(self._T, dtype=int))
        # Buffer holds the last T outputs from the bank (one per delay tap).
        self._buf = np.zeros((self._T, n_dims))

    @property
    def n_dims(self) -> int:
        return self.template.shape[1]

    def _norm(self, x: np.ndarray) -> float:
        return float(np.linalg.norm(x)) or 1.0

    def step(self, x: np.ndarray) -> dict:
        """Feed one time step of activation.  Returns match score and detection flag."""
        x   = np.asarray(x, dtype=np.float64).ravel()
        out = self._bank.push(x)  # (T,) -- one delayed value per tap
        # Shift buffer and insert new tap outputs.
        self._buf = np.roll(self._buf, -1, axis=0)
        self._buf[-1] = out

        # Normalised cross-correlation with template.
        numer = float(np.sum(self._buf * self.template))
        denom = self._norm(self._buf.ravel()) * self._norm(self.template.ravel())
        corr  = numer / denom if denom > 0 else 0.0
        detected = corr >= self.threshold
        return {"correlation": corr, "detected": detected,
                "threshold": self.threshold}

    def reset(self) -> None:
        self._bank.reset()
        self._buf[:] = 0.0
