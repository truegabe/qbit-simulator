"""Oscillatory relay -- phase-coded inter-region communication.

The brain uses neural oscillations (rhythmic activity) to selectively
route signals between regions.  The key mechanism is

    COMMUNICATION THROUGH COHERENCE  (Fries 2005, 2015):

        Two regions communicate ONLY when their oscillations are
        in phase.  When out of phase, spikes from region A arrive
        during region B's inhibitory trough -- they are ignored.

Frequency bands used for different functions:
  Gamma  (30-100 Hz) -- local feature binding, visual processing
  Beta   (12-30 Hz)  -- top-down predictions, motor preparation
  Theta  (4-12 Hz)   -- hippocampal sequences, cross-region coordination
  Alpha  (8-12 Hz)   -- sensory gating (high alpha = suppressed input)
  Delta  (1-4 Hz)    -- slow cortical potentials, sleep consolidation

How information is encoded in phase:
  PHASE MODULATION -- value v is encoded as phase offset phi = pi * v
  AMPLITUDE MODULATION -- value v modulates the carrier amplitude

This module implements:
  OscillatoryEncoder   -- real signal -> phase-modulated oscillation
  OscillatoryDecoder   -- oscillation -> reconstruct signal (Hilbert)
  PhaseCoherence       -- measure Phase Locking Value (PLV) between signals
  OscillatoryRelay     -- full encode -> sync check -> decode pipeline
  FrequencyBand        -- named preset (gamma, theta, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Frequency band presets
# ---------------------------------------------------------------------------

BANDS = {
    "delta": (1.0,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0,  12.0),
    "beta":  (12.0, 30.0),
    "gamma": (30.0, 80.0),
}


# ---------------------------------------------------------------------------
# Phase extraction (analytic signal via Hilbert transform)
# ---------------------------------------------------------------------------

def hilbert_transform(x: np.ndarray) -> np.ndarray:
    """Compute the analytic signal of x via FFT-based Hilbert transform.

    Returns complex array z where:
      np.real(z) == x
      np.imag(z) == Hilbert transform of x
      np.angle(z) == instantaneous phase
      np.abs(z)   == instantaneous amplitude (envelope)
    """
    N   = len(x)
    X   = np.fft.fft(x)
    H   = np.zeros(N)
    if N % 2 == 0:
        H[0] = H[N // 2] = 1
        H[1:N // 2] = 2
    else:
        H[0] = 1
        H[1:(N + 1) // 2] = 2
    return np.fft.ifft(X * H)


def instantaneous_phase(x: np.ndarray) -> np.ndarray:
    """Return instantaneous phase in [-pi, pi] for each sample."""
    return np.angle(hilbert_transform(x))


def phase_locking_value(phi_a: np.ndarray, phi_b: np.ndarray) -> float:
    """Phase Locking Value (PLV) between two phase series.

    PLV = |mean(exp(i * (phi_a - phi_b)))|

    PLV = 1.0  -> perfectly synchronized (constant phase difference)
    PLV = 0.0  -> no synchronization (random phase difference)
    """
    return float(np.abs(np.mean(np.exp(1j * (phi_a - phi_b)))))


# ---------------------------------------------------------------------------
# OscillatoryEncoder
# ---------------------------------------------------------------------------

@dataclass
class OscillatoryEncoder:
    """Encode a real-valued vector as a phase-modulated oscillation.

    Each dimension i of the input vector x[i] is encoded as a separate
    oscillatory channel:

        s_i(t) = A * sin(2*pi*f*t + phi_i)
        phi_i  = pi * clip(x[i], -1, 1)    (phase modulation)

    Parameters
    ----------
    n_dims      : number of input dimensions
    carrier_hz  : carrier frequency (Hz)
    sample_rate : samples per second
    n_cycles    : number of carrier cycles to generate per encoding
    amplitude   : carrier amplitude
    """
    n_dims:      int
    carrier_hz:  float = 40.0    # gamma band default
    sample_rate: float = 1000.0  # 1 kHz (1 ms bins)
    n_cycles:    float = 5.0
    amplitude:   float = 1.0

    @property
    def n_samples(self) -> int:
        return int(self.n_cycles * self.sample_rate / self.carrier_hz)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode x (n_dims,) -> oscillatory signal (n_dims, n_samples).

        Each row is one oscillatory channel.
        """
        x    = np.asarray(x, dtype=np.float64).ravel()
        T    = self.n_samples
        t    = np.arange(T) / self.sample_rate
        base = 2 * np.pi * self.carrier_hz * t   # (T,)
        # Phase modulation: phi = pi * clip(x, -1, 1)
        x_n  = np.clip(x / (np.abs(x).max() + 1e-12), -1, 1)  # normalize
        phis = np.pi * x_n                                       # (n_dims,)
        # Broadcast: (n_dims, 1) + (1, T)
        signals = self.amplitude * np.sin(phis[:, None] + base[None, :])
        return signals   # (n_dims, T)

    def carrier(self) -> np.ndarray:
        """Pure carrier with zero phase offset."""
        T = self.n_samples
        t = np.arange(T) / self.sample_rate
        return np.sin(2 * np.pi * self.carrier_hz * t)


# ---------------------------------------------------------------------------
# OscillatoryDecoder
# ---------------------------------------------------------------------------

@dataclass
class OscillatoryDecoder:
    """Decode a phase-modulated oscillation back to a real-valued vector.

    Inverts OscillatoryEncoder by extracting the mean instantaneous phase
    of each channel and inverting the phi = pi * x_norm mapping.
    """
    n_dims:      int
    carrier_hz:  float = 40.0
    sample_rate: float = 1000.0
    n_cycles:    float = 5.0

    def decode(self, signals: np.ndarray,
               x_scale: float = 1.0) -> np.ndarray:
        """Decode oscillatory signals (n_dims, T) -> x (n_dims,).

        x_scale: multiply output by this to undo normalization.
        """
        n_dims, T = signals.shape
        t       = np.arange(T) / self.sample_rate
        carrier = np.sin(2 * np.pi * self.carrier_hz * t)
        x_rec   = np.zeros(n_dims)
        for i in range(n_dims):
            # Compute phase difference between encoded signal and pure carrier.
            phi_sig     = instantaneous_phase(signals[i])
            phi_carrier = instantaneous_phase(carrier)
            dphi        = phi_sig - phi_carrier
            # Mean phase difference (circular mean).
            mean_dphi   = float(np.angle(np.mean(np.exp(1j * dphi))))
            # Invert: phi = pi * x_norm -> x_norm = phi / pi
            x_rec[i]    = mean_dphi / np.pi
        return x_rec * x_scale


# ---------------------------------------------------------------------------
# PhaseCoherence
# ---------------------------------------------------------------------------

class PhaseCoherence:
    """Measure and track phase locking between two oscillatory channels.

    Used to decide whether two regions are synchronized enough to
    communicate (PLV threshold gate).
    """

    def __init__(self, plv_threshold: float = 0.7) -> None:
        self.plv_threshold = plv_threshold
        self._history: list[float] = []

    def measure(self, sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        """Return PLV between two oscillatory signals."""
        phi_a = instantaneous_phase(sig_a)
        phi_b = instantaneous_phase(sig_b)
        plv   = phase_locking_value(phi_a, phi_b)
        self._history.append(plv)
        return plv

    def is_coherent(self, sig_a: np.ndarray, sig_b: np.ndarray) -> bool:
        return self.measure(sig_a, sig_b) >= self.plv_threshold

    def mean_plv(self) -> float:
        return float(np.mean(self._history)) if self._history else 0.0


# ---------------------------------------------------------------------------
# OscillatoryRelay
# ---------------------------------------------------------------------------

@dataclass
class OscillatoryRelay:
    """Full oscillatory encode -> coherence check -> decode pipeline.

    Protocol:
      1. Sender encodes x as phase-modulated oscillation (n_dims channels).
      2. A reference carrier is broadcast to both sides (like a common clock).
      3. Receiver checks PLV between incoming signal and its local carrier.
         If PLV >= plv_threshold: coherent -> decode and pass.
         If PLV <  plv_threshold: incoherent -> block (return zeros).
      4. Receiver decodes the phase offsets back to x.

    Parameters
    ----------
    n_dims          : signal dimensionality
    band            : frequency band name ('gamma', 'theta', etc.) OR
                      explicit carrier_hz float
    sample_rate     : samples per second (default 1000 = 1 ms bins)
    n_cycles        : carrier cycles per packet
    plv_threshold   : minimum PLV to allow communication
    phase_noise_std : std of Gaussian phase noise added to transmitted signal
                      (models realistic synchronization jitter)
    """
    n_dims:           int
    band:             str | float = "gamma"
    sample_rate:      float       = 1000.0
    n_cycles:         float       = 5.0
    plv_threshold:    float       = 0.7
    phase_noise_std:  float       = 0.0
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    encoder:   OscillatoryEncoder  = field(default=None, repr=False)
    decoder:   OscillatoryDecoder  = field(default=None, repr=False)
    coherence: PhaseCoherence      = field(default=None, repr=False)
    _x_scale:  float               = field(default=1.0, repr=False)

    def __post_init__(self) -> None:
        hz = BANDS[self.band][0] if isinstance(self.band, str) else float(self.band)
        self.encoder   = OscillatoryEncoder(
            self.n_dims, carrier_hz=hz,
            sample_rate=self.sample_rate, n_cycles=self.n_cycles)
        self.decoder   = OscillatoryDecoder(
            self.n_dims, carrier_hz=hz,
            sample_rate=self.sample_rate, n_cycles=self.n_cycles)
        self.coherence = PhaseCoherence(self.plv_threshold)

    def transmit(self, x: np.ndarray,
                 force_coherent: bool = False) -> tuple[np.ndarray, dict]:
        """Encode, optionally add noise, check coherence, decode.

        Returns
        -------
        x_rec  : reconstructed signal (zeros if incoherent and not forced)
        stats  : dict with plv, coherent, x_scale, reconstruction_error
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        self._x_scale = float(np.abs(x).max()) or 1.0

        # Encode.
        signals = self.encoder.encode(x)   # (n_dims, T)

        # Add phase noise if requested.
        if self.phase_noise_std > 0:
            noise_phase = self.rng.normal(0, self.phase_noise_std,
                                           signals.shape)
            t   = np.arange(signals.shape[1]) / self.sample_rate
            hz  = self.encoder.carrier_hz
            carrier_arg = 2 * np.pi * hz * t
            # Rotate signal by noise phase: s*cos(n) + carrier*sin(n)
            signals = (signals * np.cos(noise_phase) +
                       self.encoder.amplitude *
                       np.sin(carrier_arg)[None, :] * np.sin(noise_phase))

        # Coherence check on channel 0 vs pure carrier.
        carrier = self.encoder.carrier()
        plv = self.coherence.measure(signals[0], carrier)
        coherent = plv >= self.plv_threshold or force_coherent

        if not coherent:
            x_rec = np.zeros(self.n_dims)
        else:
            x_rec = self.decoder.decode(signals, x_scale=self._x_scale)

        stats = {
            "plv":                  plv,
            "coherent":             coherent,
            "reconstruction_error": float(
                np.linalg.norm(x - x_rec) / (np.linalg.norm(x) + 1e-12)),
            "band": self.band,
            "carrier_hz": self.encoder.carrier_hz,
        }
        return x_rec, stats

    def synchronize(self) -> None:
        """Reset phase coherence history (re-establish sync)."""
        self.coherence._history.clear()

    def mean_plv(self) -> float:
        return self.coherence.mean_plv()
