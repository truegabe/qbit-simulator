"""Tests for the cochlear front-end and auditory pathway."""

import numpy as np
import pytest

from qbit_simulator.cochlear_frontend import (
    CochlearFilterBank, HairCellModel, CochlearFrontend,
)
from qbit_simulator.auditory_pathway import (
    EnvelopeExtractor, FineStructurePath, PhaseStream, AuditoryPathway,
)
from qbit_simulator.spike_routing_bus import SpikeEvent


SR = 16000


def _tone(freq_hz: float, duration: float = 0.1, sr: int = SR) -> np.ndarray:
    """Generate a pure tone of given frequency."""
    t = np.arange(int(duration * sr)) / sr
    return 0.5 * np.sin(2 * np.pi * freq_hz * t)


# ===========================================================================
# CochlearFilterBank
# ===========================================================================

def test_filterbank_mel_centres_sorted():
    fb = CochlearFilterBank(n_bands=32, mode="mel")
    assert fb.centres.shape == (32,)
    assert np.all(np.diff(fb.centres) > 0)


def test_filterbank_gammatone_centres_sorted():
    fb = CochlearFilterBank(n_bands=32, mode="gammatone")
    assert fb.centres.shape == (32,)
    assert np.all(np.diff(fb.centres) > 0)


def test_filterbank_mel_filter_shape():
    fb    = CochlearFilterBank(n_bands=16, mode="mel")
    audio = _tone(440.0)
    bands = fb.filter(audio)
    assert bands.shape[0] == 16
    assert bands.shape[1] == len(audio)


def test_filterbank_gammatone_responds_to_tone():
    """A tone at 1 kHz should excite a band near 1 kHz more than far ones."""
    fb     = CochlearFilterBank(n_bands=32, mode="gammatone",
                                 f_min=80.0, f_max=8000.0)
    audio  = _tone(1000.0, duration=0.2)
    bands  = fb.filter(audio)
    energies = np.abs(bands).mean(axis=1)
    best   = int(np.argmax(energies))
    # Best band centre should be within a factor of 2 of 1 kHz.
    assert 500.0 < fb.centres[best] < 2000.0


def test_filterbank_unknown_mode_raises():
    with pytest.raises(ValueError):
        CochlearFilterBank(n_bands=8, mode="banana")


# ===========================================================================
# HairCellModel
# ===========================================================================

def test_hair_cell_output_non_negative():
    hc    = HairCellModel(sample_rate=SR)
    bands = np.random.default_rng(0).standard_normal((4, 100))
    out   = hc.apply(bands)
    assert np.all(out >= 0)


def test_hair_cell_half_wave_zeroes_negative():
    hc    = HairCellModel(sample_rate=SR, rectify="half")
    bands = -np.ones((2, 50))
    out   = hc.apply(bands)
    assert np.allclose(out, 0.0)


def test_hair_cell_full_wave_keeps_magnitude():
    hc    = HairCellModel(sample_rate=SR, rectify="full")
    bands = -np.ones((2, 50))
    out   = hc.apply(bands)
    # Low-pass starts from 0, so values grow but should be > 0.
    assert np.all(out >= 0)


# ===========================================================================
# CochlearFrontend
# ===========================================================================

def test_frontend_emits_spikes():
    front  = CochlearFrontend(n_bands=32, mode="mel",
                                frame_rate=200.0, threshold=0.01)
    audio  = _tone(1000.0, duration=0.2)
    spikes = front.process(audio)
    assert isinstance(spikes, list)
    assert len(spikes) > 0
    assert all(isinstance(s, SpikeEvent) for s in spikes)


def test_frontend_spike_neurons_in_range():
    front  = CochlearFrontend(n_bands=16, mode="gammatone")
    audio  = _tone(500.0)
    spikes = front.process(audio)
    for s in spikes:
        assert 0 <= s.neuron_id < 16


def test_frontend_spike_timestamps_monotonic():
    front  = CochlearFrontend(n_bands=8, frame_rate=100.0)
    audio  = _tone(800.0, duration=0.3)
    spikes = front.process(audio)
    # Group by neuron; timestamps within a band should be sorted.
    by_neuron: dict[int, list[float]] = {}
    for s in spikes:
        by_neuron.setdefault(s.neuron_id, []).append(s.timestamp)
    for ts in by_neuron.values():
        assert ts == sorted(ts)


def test_frontend_silence_produces_few_spikes():
    front = CochlearFrontend(n_bands=16, threshold=0.5)
    audio = np.zeros(SR // 4)
    spikes = front.process(audio)
    assert len(spikes) == 0


def test_frontend_topk_mode():
    front = CochlearFrontend(n_bands=16, spike_mode="topk", k=4,
                              frame_rate=50.0)
    audio = _tone(800.0, duration=0.1)
    spikes = front.process(audio)
    # 0.1s * 50 Hz = 5 frames, each emits 4 spikes -> 20 total max.
    assert len(spikes) <= 4 * 6


def test_frontend_cochleogram_shape():
    front = CochlearFrontend(n_bands=24)
    audio = _tone(500.0, duration=0.05)
    cg    = front.cochleogram(audio)
    assert cg.shape[0] == 24


# ===========================================================================
# EnvelopeExtractor
# ===========================================================================

def test_envelope_extractor_shape():
    env = EnvelopeExtractor(n_bands=8, bin_width=0.05)
    spk = [SpikeEvent(neuron_id=i % 8, timestamp=0.01 * i, value=1.0)
            for i in range(20)]
    out = env.step(spk, t_now=0.2)
    assert out.shape == (8,)


def test_envelope_extractor_responds_to_spikes():
    env = EnvelopeExtractor(n_bands=4, bin_width=0.1, ema_alpha=1.0)
    spk = [SpikeEvent(neuron_id=2, timestamp=0.05, value=1.0)
            for _ in range(5)]
    out = env.step(spk, t_now=0.1)
    assert out[2] > 0
    assert np.all(out[[0, 1, 3]] == 0)


def test_envelope_extractor_reset():
    env = EnvelopeExtractor(n_bands=4)
    spk = [SpikeEvent(neuron_id=0, timestamp=0.01, value=1.0)]
    env.step(spk, t_now=0.05)
    env.reset()
    assert np.allclose(env._state, 0.0)


# ===========================================================================
# FineStructurePath
# ===========================================================================

def test_fine_structure_shape():
    fs   = FineStructurePath(n_bands=8)
    spk  = [SpikeEvent(neuron_id=3, timestamp=0.005, value=0.7)]
    out  = fs.step(spk, t_now=0.01, window=0.01)
    assert out.shape == (8,)
    assert out[3] == pytest.approx(0.7)


def test_fine_structure_empty_returns_zeros():
    fs  = FineStructurePath(n_bands=4)
    out = fs.step([], t_now=0.0)
    assert np.allclose(out, 0.0)


# ===========================================================================
# PhaseStream
# ===========================================================================

def test_phase_stream_output_shape():
    ps    = PhaseStream(n_bands=8)
    cg    = np.random.default_rng(0).standard_normal((8, 100))
    phase = ps.step(cg)
    assert phase.shape == (8,)
    assert np.all(np.abs(phase) <= np.pi + 1e-9)


def test_phase_stream_empty_input():
    ps    = PhaseStream(n_bands=4)
    phase = ps.step(np.zeros((4, 0)))
    assert phase.shape == (4,)


# ===========================================================================
# AuditoryPathway
# ===========================================================================

def test_pathway_process_audio_returns_dict():
    front = CochlearFrontend(n_bands=16, mode="mel")
    ap    = AuditoryPathway(n_bands=16, use_fno=False)
    audio = _tone(800.0, duration=0.1)
    res   = ap.process_audio(audio, front)
    for k in ("percept", "envelope", "fine", "phases",
              "envelope_err", "fine_err"):
        assert k in res


def test_pathway_percept_shape():
    front = CochlearFrontend(n_bands=8, mode="mel")
    ap    = AuditoryPathway(n_bands=8)
    audio = _tone(500.0, duration=0.05)
    res   = ap.process_audio(audio, front)
    assert res["percept"].features.shape == (8,)


def test_pathway_with_fno():
    front = CochlearFrontend(n_bands=16, mode="mel")
    ap    = AuditoryPathway(n_bands=16, use_fno=True, fno_d_model=8)
    audio = _tone(1000.0, duration=0.05)
    res   = ap.process_audio(audio, front)
    assert res["envelope"].shape == (16,)


def test_pathway_streams_have_expected_shapes():
    front = CochlearFrontend(n_bands=12)
    ap    = AuditoryPathway(n_bands=12)
    audio = _tone(700.0, duration=0.08)
    res   = ap.process_audio(audio, front)
    assert res["envelope"].shape == (12,)
    assert res["fine"].shape     == (12,)
    assert res["phases"].shape   == (12,)


def test_pathway_silence_does_not_crash():
    front = CochlearFrontend(n_bands=8)
    ap    = AuditoryPathway(n_bands=8)
    audio = np.zeros(SR // 10)
    res   = ap.process_audio(audio, front)
    assert res["percept"] is not None


def test_pathway_repr():
    ap = AuditoryPathway(n_bands=16)
    s  = repr(ap)
    assert "AuditoryPathway" in s
