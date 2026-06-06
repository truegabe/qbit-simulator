"""Auditory pathway -- spikes to unified percept.

Takes the spike stream from CochlearFrontend and runs it through a model
of the central auditory system:

  spikes (from cochlea)
      |
      +-- ENVELOPE PATH ----------+
      |   (slow ~16 Hz syllable     -- PredictiveRelay
      |    rate, captures rhythm)      tracks slow dynamics
      |
      +-- FINE-STRUCTURE PATH ----+
      |   (fast carrier, captures   -- OscillatoryRelay
      |    formants / vowel ID)        rides on gamma phase
      |
      +-- PHASE STREAM -----------+
      |   (Hilbert phase per band)  -- raw, fed to BindingBus
      |
      +-- OPTIONAL FNO REFINEMENT--+
          (learnable post-filter)   -- FNO1d cleans spike->percept map
                                       BEFORE binding
                  v
              BINDING BUS
              PLV-synchronises envelope + carrier
                  v
          UNIFIED AUDITORY PERCEPT
          (features, binding_mask, mean_plv, region_names)

Classes
-------
  EnvelopeExtractor  -- slow rate path from spike stream
  FineStructurePath  -- fast carrier path
  PhaseStream        -- per-band Hilbert phase
  AuditoryPathway    -- full pipeline orchestrator
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from qbit_simulator.spike_routing_bus import (
    SpikeEvent, AERDecoder,
)
from qbit_simulator.predictive_relay import PredictiveRelay
from qbit_simulator.oscillatory_relay import OscillatoryRelay
from qbit_simulator.binding_bus       import BindingBus, FeatureBundle
from qbit_simulator.fno_core          import FNO1d


# ---------------------------------------------------------------------------
# EnvelopeExtractor
# ---------------------------------------------------------------------------

class EnvelopeExtractor:
    """Slow-envelope path: bin spike rates into syllable-rate frames.

    Aggregates spikes into ~50 ms windows (= 20 Hz frame rate) and
    smooths with EMA to track speech envelope.

    Parameters
    ----------
    n_bands     : cochlear band count (= input neuron count)
    bin_width   : aggregation window in seconds
    ema_alpha   : exponential smoothing factor (0..1)
    """

    def __init__(self, n_bands: int = 64,
                 bin_width: float = 0.05,
                 ema_alpha: float = 0.3) -> None:
        self.n_bands   = n_bands
        self.bin_width = bin_width
        self.ema_alpha = ema_alpha
        self._state    = np.zeros(n_bands)

    def step(self, spikes: list[SpikeEvent],
             t_now: float) -> np.ndarray:
        """Aggregate spikes within [t_now - bin_width, t_now] and update EMA.

        Returns
        -------
        envelope : (n_bands,) smoothed firing rate
        """
        t_lo  = t_now - self.bin_width
        rates = np.zeros(self.n_bands)
        for e in spikes:
            if t_lo <= e.timestamp <= t_now and 0 <= e.neuron_id < self.n_bands:
                rates[e.neuron_id] += abs(e.value)
        rates /= max(self.bin_width, 1e-6)        # convert count -> Hz

        a = self.ema_alpha
        self._state = a * rates + (1 - a) * self._state
        return self._state.copy()

    def reset(self) -> None:
        self._state[:] = 0.0


# ---------------------------------------------------------------------------
# FineStructurePath
# ---------------------------------------------------------------------------

class FineStructurePath:
    """Fast-carrier path: keeps the per-spike timing and intensity.

    Decodes recent spikes into a dense per-band signal at a higher
    rate than the envelope path.  This preserves formant transitions
    (the things that distinguish vowels and consonants).

    Parameters
    ----------
    n_bands   : cochlear band count
    decoder   : decoder mode for AERDecoder ('latest' | 'sum' | 'mean')
    """

    def __init__(self, n_bands: int = 64, decoder: str = "latest") -> None:
        self.n_bands = n_bands
        self.decoder = AERDecoder(n_neurons=n_bands, mode=decoder)

    def step(self, spikes: list[SpikeEvent],
             t_now: float, window: float = 0.01) -> np.ndarray:
        """Decode recent spikes (last `window` seconds) into a dense vector."""
        recent = [s for s in spikes if t_now - window <= s.timestamp <= t_now]
        return self.decoder.decode(recent)


# ---------------------------------------------------------------------------
# PhaseStream
# ---------------------------------------------------------------------------

class PhaseStream:
    """Per-band instantaneous phase from a cochleogram window.

    Uses Hilbert transform along the time axis to get analytic signal
    and extracts phase.  Returns one phase value per band (the latest
    in the window).

    Parameters
    ----------
    n_bands : cochlear band count
    """

    def __init__(self, n_bands: int = 64) -> None:
        self.n_bands = n_bands

    def step(self, cochleogram: np.ndarray) -> np.ndarray:
        """Compute the latest instantaneous phase per band.

        Parameters
        ----------
        cochleogram : (n_bands, n_samples)

        Returns
        -------
        phases : (n_bands,) in [-pi, pi]
        """
        if cochleogram.size == 0:
            return np.zeros(self.n_bands)

        # Analytic signal via FFT (avoids scipy dependency).
        n      = cochleogram.shape[1]
        X      = np.fft.fft(cochleogram, axis=1)
        h      = np.zeros(n)
        if n % 2 == 0:
            h[0] = h[n // 2] = 1
            h[1:n // 2]      = 2
        else:
            h[0]                 = 1
            h[1:(n + 1) // 2]    = 2
        analytic = np.fft.ifft(X * h[None, :], axis=1)
        # Latest phase per band.
        return np.angle(analytic[:, -1])


# ---------------------------------------------------------------------------
# AuditoryPathway
# ---------------------------------------------------------------------------

@dataclass
class AuditoryPathway:
    """Full spikes-or-audio -> unified-percept pipeline.

    Usage
    -----
        ap = AuditoryPathway(n_bands=64)
        # If you have raw audio:
        percept = ap.process_audio(audio_waveform, cochlear_frontend)
        # If you already have spikes + cochleogram:
        percept = ap.process_spikes(spikes, cochleogram, t_now=...)

    Parameters
    ----------
    n_bands         : cochlear band count
    envelope_bins   : bin width for envelope path (s)
    fine_window     : window for fine-structure path (s)
    use_fno         : if True, run FNO refinement before binding
    fno_d_model     : FNO width (only used if use_fno=True)
    plv_threshold   : binding threshold
    """
    n_bands:        int   = 64
    envelope_bins:  float = 0.05
    fine_window:    float = 0.01
    use_fno:        bool  = False
    fno_d_model:    int   = 16
    plv_threshold:  float = 0.5
    seed:           int   = 0

    env:        EnvelopeExtractor = field(default=None, repr=False)
    fine:       FineStructurePath = field(default=None, repr=False)
    phase:      PhaseStream       = field(default=None, repr=False)
    pred_relay: PredictiveRelay   = field(default=None, repr=False)
    osc_relay:  OscillatoryRelay  = field(default=None, repr=False)
    bus:        BindingBus        = field(default=None, repr=False)
    fno:        Optional[FNO1d]   = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.env        = EnvelopeExtractor(self.n_bands, self.envelope_bins)
        self.fine       = FineStructurePath(self.n_bands)
        self.phase      = PhaseStream(self.n_bands)
        self.pred_relay = PredictiveRelay(
            n_dims=self.n_bands, predictor_type="ema")
        self.osc_relay  = OscillatoryRelay(
            n_dims=self.n_bands, band="gamma")
        self.bus        = BindingBus(
            plv_threshold=self.plv_threshold,
            phase_noise_std=0.05, merge_mode="weighted",
            output_dim=self.n_bands)
        self.bus.register("envelope", self.n_bands)
        self.bus.register("fine",     self.n_bands)

        if self.use_fno:
            self.fno = FNO1d(
                d_in=1, d_out=1, d_model=self.fno_d_model,
                n_layers=2,
                k_max=min(8, self.n_bands // 2 + 1),
                seed=self.seed)

    # ------------------------------------------------------------------
    def process_spikes(self, spikes: list[SpikeEvent],
                        cochleogram: np.ndarray,
                        t_now: float,
                        rng: Optional[np.random.Generator] = None) -> dict:
        """Run the full pathway on a chunk of spikes + cochleogram.

        Returns
        -------
        result : dict with
            "percept"      : UnifiedPercept
            "envelope"     : (n_bands,) slow rate
            "fine"         : (n_bands,) fast carrier
            "phases"       : (n_bands,) per-band phase
            "envelope_err" : reconstruction error from PredictiveRelay
            "fine_err"     : reconstruction error from OscillatoryRelay
        """
        rng = rng or np.random.default_rng(self.seed)

        # 1. Three parallel streams.
        envelope = self.env.step(spikes, t_now)
        fine     = self.fine.step(spikes, t_now, window=self.fine_window)
        phases   = self.phase.step(cochleogram)

        # 2. Run each through its relay.
        env_rec,   env_stats = self.pred_relay.relay(envelope)
        fine_rec,  osc_stats = self.osc_relay.transmit(fine)

        # 3. Optional FNO refinement -- learns to clean spike->percept residual.
        if self.fno is not None:
            env_rec  = self._fno_refine(env_rec)
            fine_rec = self._fno_refine(fine_rec)

        # 4. Wrap into FeatureBundles with the actual phases as tags.
        env_bundle = FeatureBundle(
            features=env_rec, phases=phases, region="envelope", timestamp=t_now)
        fine_bundle = FeatureBundle(
            features=fine_rec, phases=phases, region="fine", timestamp=t_now)

        # 5. Bind.
        percept = self.bus.bind([env_bundle, fine_bundle])

        return {
            "percept":      percept,
            "envelope":     envelope,
            "fine":         fine,
            "phases":       phases,
            "envelope_err": env_stats.get("relay_error",
                              env_stats.get("reconstruction_error", 0.0)),
            "fine_err":     osc_stats.get("reconstruction_error", 0.0),
        }

    def process_audio(self, audio: np.ndarray, frontend,
                      t_offset: float = 0.0,
                      rng: Optional[np.random.Generator] = None) -> dict:
        """Convenience: take raw audio + a CochlearFrontend, return percept.

        Parameters
        ----------
        audio      : raw waveform
        frontend   : CochlearFrontend instance
        t_offset   : timestamp of audio[0]
        """
        spikes      = frontend.process(audio, t_offset=t_offset)
        cochleogram = frontend.cochleogram(audio)
        t_now       = t_offset + len(audio) / frontend.sample_rate
        return self.process_spikes(spikes, cochleogram, t_now, rng=rng)

    # ------------------------------------------------------------------
    def _fno_refine(self, x: np.ndarray) -> np.ndarray:
        """Apply 1-channel FNO refinement along the band axis."""
        x_in  = np.asarray(x, dtype=np.float64).ravel()[:, np.newaxis]
        out   = self.fno.forward(x_in)[:, 0]
        return out

    def reset(self) -> None:
        """Clear all internal state."""
        self.env.reset()

    def __repr__(self) -> str:
        return (f"AuditoryPathway(bands={self.n_bands}, "
                f"use_fno={self.use_fno})")
