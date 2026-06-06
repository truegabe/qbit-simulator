"""Cochlear front-end -- audio waveform to spike train.

Mirrors the human ear + brainstem pathway:

  audio (waveform)
      |
      v
  COCHLEAR FILTER BANK   (basilar membrane: frequency decomposition)
      - 'mel'       : log-spaced filter centers (simple, fast)
      - 'gammatone' : matches real cochlear hair-cell tuning curves
      |
      v
  HAIR CELL MODEL        (inner hair cells: rectify + low-pass)
      - half-wave rectification
      - low-pass smoothing (~1 kHz cutoff)
      |
      v
  TONOTOPIC SPIKE ENCODER (auditory nerve: rate-coded spikes)
      - feeds AEREncoder per-band activations at frame_rate Hz
      - output: SpikeEvent stream with (band_id, time, intensity)

The output stream is directly compatible with SpikeRoutingBus and
all downstream brain regions that consume spikes.

Classes
-------
  CochlearFilterBank   -- mel or gammatone filter bank
  HairCellModel        -- rectify + low-pass
  CochlearFrontend     -- full audio -> spike pipeline
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from qbit_simulator.spike_routing_bus import (
    SpikeEvent, AEREncoder,
)


# ---------------------------------------------------------------------------
# CochlearFilterBank
# ---------------------------------------------------------------------------

class CochlearFilterBank:
    """Frequency decomposition of audio into n_bands cochlear channels.

    Parameters
    ----------
    n_bands     : number of frequency bands (cochlear channels)
    sample_rate : audio sample rate in Hz
    f_min       : lowest band centre (Hz)
    f_max       : highest band centre (Hz)
    mode        : 'mel' or 'gammatone'
    """

    def __init__(self, n_bands: int = 64, sample_rate: int = 16000,
                 f_min: float = 80.0, f_max: float = 8000.0,
                 mode: str = "gammatone") -> None:
        self.n_bands     = n_bands
        self.sample_rate = sample_rate
        self.f_min       = f_min
        self.f_max       = min(f_max, sample_rate // 2 - 1)
        self.mode        = mode

        self.centres = self._make_centres()
        if mode == "gammatone":
            self._impulse_responses = self._gammatone_impulses()
        else:
            self._impulse_responses = None   # mel uses FFT path

    # ------------------------------------------------------------------
    @staticmethod
    def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _make_centres(self) -> np.ndarray:
        """Band centre frequencies (Hz), spaced according to mode."""
        if self.mode == "mel":
            m_min = self._hz_to_mel(np.array(self.f_min))
            m_max = self._hz_to_mel(np.array(self.f_max))
            mels  = np.linspace(m_min, m_max, self.n_bands)
            return self._mel_to_hz(mels)

        if self.mode == "gammatone":
            # ERB scale (Equivalent Rectangular Bandwidth) -- cochlear-accurate.
            erb_min = self._hz_to_erb(self.f_min)
            erb_max = self._hz_to_erb(self.f_max)
            erbs    = np.linspace(erb_min, erb_max, self.n_bands)
            return self._erb_to_hz(erbs)

        raise ValueError(f"unknown mode: {self.mode}")

    @staticmethod
    def _hz_to_erb(hz: float) -> float:
        return 21.4 * np.log10(0.00437 * hz + 1.0)

    @staticmethod
    def _erb_to_hz(erb: np.ndarray) -> np.ndarray:
        return (10.0 ** (erb / 21.4) - 1.0) / 0.00437

    # ------------------------------------------------------------------
    def _gammatone_impulses(self, duration: float = 0.025) -> np.ndarray:
        """Build gammatone impulse responses for all bands."""
        n_samples = int(duration * self.sample_rate)
        t         = np.arange(n_samples) / self.sample_rate
        order     = 4
        responses = np.zeros((self.n_bands, n_samples))

        for i, fc in enumerate(self.centres):
            erb_bw = 24.7 * (4.37e-3 * fc + 1.0)
            b      = 1.019 * 2 * np.pi * erb_bw
            ir     = (t ** (order - 1)) * np.exp(-b * t) * np.cos(2 * np.pi * fc * t)
            norm   = np.sum(np.abs(ir)) or 1.0
            responses[i] = ir / norm

        return responses

    # ------------------------------------------------------------------
    def filter(self, audio: np.ndarray) -> np.ndarray:
        """Decompose audio into per-band signals.

        Parameters
        ----------
        audio : (n_samples,) raw waveform, normalised to ~[-1, 1]

        Returns
        -------
        bands : (n_bands, n_samples) per-channel filtered signal
        """
        audio = np.asarray(audio, dtype=np.float64).ravel()

        if self.mode == "gammatone":
            return self._filter_gammatone(audio)
        return self._filter_mel(audio)

    def _filter_gammatone(self, audio: np.ndarray) -> np.ndarray:
        n      = len(audio)
        bands  = np.zeros((self.n_bands, n))
        for i, ir in enumerate(self._impulse_responses):
            # Use FFT convolution for speed.
            conv      = np.fft.irfft(
                np.fft.rfft(audio, n=n + len(ir) - 1) *
                np.fft.rfft(ir,    n=n + len(ir) - 1))
            bands[i]  = conv[:n]
        return bands

    def _filter_mel(self, audio: np.ndarray) -> np.ndarray:
        # STFT-based mel filtering.
        win_len = min(512, len(audio))
        hop     = max(win_len // 4, 1)
        n_fft   = 1 << (win_len - 1).bit_length()

        # Build mel filter weights once.
        freqs   = np.fft.rfftfreq(n_fft, 1.0 / self.sample_rate)
        weights = self._mel_filter_weights(freqs)

        # Frame the signal.
        n_frames = max(1, (len(audio) - win_len) // hop + 1)
        bands    = np.zeros((self.n_bands, len(audio)))
        window   = np.hanning(win_len)

        for f in range(n_frames):
            start = f * hop
            stop  = start + win_len
            frame = audio[start:stop] * window[:len(audio[start:stop])]
            if len(frame) < win_len:
                frame = np.pad(frame, (0, win_len - len(frame)))
            spec = np.abs(np.fft.rfft(frame, n=n_fft))
            mel  = weights @ spec        # (n_bands,)
            # Fill the frame in band signals with this energy (constant per frame).
            bands[:, start:stop] = mel[:, None]

        return bands

    def _mel_filter_weights(self, freqs: np.ndarray) -> np.ndarray:
        """Triangular mel filter weights: (n_bands, len(freqs))."""
        centres = self.centres
        edges   = np.concatenate([
            [self.f_min],
            (centres[:-1] + centres[1:]) / 2,
            [self.f_max],
        ])
        weights = np.zeros((self.n_bands, len(freqs)))
        for i in range(self.n_bands):
            lo, ctr, hi = edges[i], centres[i], edges[i + 1]
            up   = (freqs - lo)  / max(ctr - lo, 1e-6)
            down = (hi - freqs)  / max(hi - ctr, 1e-6)
            weights[i] = np.maximum(0.0, np.minimum(up, down))
        return weights


# ---------------------------------------------------------------------------
# HairCellModel
# ---------------------------------------------------------------------------

class HairCellModel:
    """Simulates inner hair cells: rectify + low-pass.

    Real hair cells fire in one direction and integrate up to ~1 kHz.
    Models this as: half-wave rectification + first-order low-pass filter.

    Parameters
    ----------
    sample_rate : Hz
    lp_cutoff   : low-pass cutoff frequency (Hz), default 1000
    rectify     : 'half' (biological) or 'full' (more sensitive)
    """

    def __init__(self, sample_rate: int = 16000,
                 lp_cutoff: float = 1000.0,
                 rectify: str = "half") -> None:
        self.sample_rate = sample_rate
        self.lp_cutoff   = lp_cutoff
        self.rectify     = rectify

        # First-order LP filter coefficient.
        # y[n] = (1 - a) * x[n] + a * y[n - 1]
        rc      = 1.0 / (2 * np.pi * lp_cutoff)
        dt      = 1.0 / sample_rate
        self.a  = rc / (rc + dt)

    def apply(self, bands: np.ndarray) -> np.ndarray:
        """Apply hair-cell model to filter-bank output.

        Parameters
        ----------
        bands : (n_bands, n_samples)

        Returns
        -------
        firing_rate : (n_bands, n_samples) non-negative
        """
        if self.rectify == "half":
            x = np.maximum(bands, 0.0)
        else:
            x = np.abs(bands)

        # First-order IIR low-pass per band.
        y = np.zeros_like(x)
        if x.shape[1] > 0:
            y[:, 0] = (1 - self.a) * x[:, 0]
            for n in range(1, x.shape[1]):
                y[:, n] = (1 - self.a) * x[:, n] + self.a * y[:, n - 1]

        return y


# ---------------------------------------------------------------------------
# CochlearFrontend
# ---------------------------------------------------------------------------

@dataclass
class CochlearFrontend:
    """Full audio -> spike train pipeline.

    Usage
    -----
        front  = CochlearFrontend(n_bands=64, sample_rate=16000)
        spikes = front.process(audio_waveform)
        # spikes is a list[SpikeEvent] (band_id, timestamp, intensity)

        # Feed into SpikeRoutingBus:
        bus.emit_events(spikes)

    Parameters
    ----------
    n_bands       : cochlear channels (= n_neurons in the auditory nerve)
    sample_rate   : audio sample rate (Hz)
    f_min, f_max  : cochlear frequency range
    mode          : 'mel' or 'gammatone'
    frame_rate    : how often to emit spikes (Hz). 200 Hz = every 5 ms.
    spike_mode    : 'rate' | 'threshold' | 'topk'
    threshold     : intensity threshold (used in 'threshold' mode)
    k             : top-k count (used in 'topk' mode)
    lp_cutoff     : hair-cell low-pass cutoff
    """
    n_bands:     int   = 64
    sample_rate: int   = 16000
    f_min:       float = 80.0
    f_max:       float = 8000.0
    mode:        str   = "gammatone"
    frame_rate:  float = 200.0
    spike_mode:  str   = "threshold"
    threshold:   float = 0.05
    k:           int   = 8
    lp_cutoff:   float = 1000.0
    seed:        int   = 0

    filters:    CochlearFilterBank = field(default=None, repr=False)
    hair_cells: HairCellModel      = field(default=None, repr=False)
    encoder:    AEREncoder         = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.filters = CochlearFilterBank(
            n_bands=self.n_bands, sample_rate=self.sample_rate,
            f_min=self.f_min, f_max=self.f_max, mode=self.mode)
        self.hair_cells = HairCellModel(
            sample_rate=self.sample_rate, lp_cutoff=self.lp_cutoff)
        self.encoder = AEREncoder(
            n_neurons=self.n_bands, mode=self.spike_mode,
            threshold=self.threshold, k=self.k,
            rng=np.random.default_rng(self.seed))

    # ------------------------------------------------------------------
    def process(self, audio: np.ndarray,
                t_offset: float = 0.0) -> list[SpikeEvent]:
        """Convert an audio waveform into a list of SpikeEvents.

        Parameters
        ----------
        audio    : (n_samples,) raw waveform in roughly [-1, 1]
        t_offset : timestamp of audio sample 0 (for streaming use)

        Returns
        -------
        spikes : list of SpikeEvent(neuron_id=band, timestamp, value)
        """
        audio   = np.asarray(audio, dtype=np.float64).ravel()
        bands   = self.filters.filter(audio)             # (n_bands, n_samples)
        rates   = self.hair_cells.apply(bands)           # (n_bands, n_samples)

        # Normalise per call so encoder thresholds make sense.
        peak = float(rates.max()) or 1.0
        rates_norm = rates / peak

        # Sample rates at frame_rate to produce spike emissions.
        samples_per_frame = max(1, int(self.sample_rate / self.frame_rate))
        n_frames          = max(1, rates_norm.shape[1] // samples_per_frame)

        spikes: list[SpikeEvent] = []
        for f in range(n_frames):
            start = f * samples_per_frame
            stop  = start + samples_per_frame
            chunk = rates_norm[:, start:stop]
            x     = chunk.mean(axis=1)            # (n_bands,)
            t     = t_offset + start / self.sample_rate
            spikes.extend(self.encoder.encode(x, timestamp=t))
        return spikes

    # ------------------------------------------------------------------
    def cochleogram(self, audio: np.ndarray) -> np.ndarray:
        """Return the (n_bands, n_samples) firing-rate cochleogram.

        Useful for visualisation / debugging without spiking.
        """
        bands = self.filters.filter(audio)
        return self.hair_cells.apply(bands)

    def __repr__(self) -> str:
        return (f"CochlearFrontend(bands={self.n_bands}, "
                f"mode='{self.mode}', sr={self.sample_rate}Hz, "
                f"frame_rate={self.frame_rate}Hz)")
