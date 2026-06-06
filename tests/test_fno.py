"""Tests for FNO core and FNO relay modules."""

import numpy as np
import pytest

from qbit_simulator.fno_core import SpectralConv1d, FNOBlock, FNO1d
from qbit_simulator.fno_relays import (
    FNOPredictiveRelay, FNOOscillatoryRelay,
    FNOHierarchicalRelay, FNODecoherenceModel,
)


# ===========================================================================
# SpectralConv1d
# ===========================================================================

def test_spectral_conv_output_shape():
    sc  = SpectralConv1d(d_in=4, d_out=8, k_max=3)
    x   = np.random.default_rng(0).standard_normal((16, 4))
    out = sc.forward(x)
    assert out.shape == (16, 8)


def test_spectral_conv_different_n_points():
    sc = SpectralConv1d(d_in=4, d_out=4, k_max=3)
    for n in [8, 16, 32, 64]:
        out = sc.forward(np.ones((n, 4)))
        assert out.shape == (n, 4)


def test_spectral_conv_param_count():
    sc = SpectralConv1d(d_in=4, d_out=8, k_max=3)
    assert sc.param_count() == 2 * 3 * 4 * 8


def test_spectral_conv_state_dict_roundtrip():
    sc  = SpectralConv1d(d_in=4, d_out=4, k_max=2)
    sd  = sc.state_dict()
    sc2 = SpectralConv1d(d_in=4, d_out=4, k_max=2, seed=99)
    sc2.load_state_dict(sd)
    x   = np.random.default_rng(0).standard_normal((8, 4))
    assert np.allclose(sc.forward(x), sc2.forward(x))


# ===========================================================================
# FNOBlock
# ===========================================================================

def test_fno_block_output_shape():
    block = FNOBlock(d_model=16, k_max=4)
    x     = np.random.default_rng(0).standard_normal((32, 16))
    out   = block.forward(x)
    assert out.shape == (32, 16)


def test_fno_block_relu():
    block = FNOBlock(d_model=8, k_max=3, activation="relu")
    x     = np.random.default_rng(0).standard_normal((16, 8))
    out   = block.forward(x)
    assert out.shape == (16, 8)


def test_fno_block_tanh():
    block = FNOBlock(d_model=8, k_max=3, activation="tanh")
    x     = np.random.default_rng(0).standard_normal((16, 8))
    out   = block.forward(x)
    assert np.all(np.abs(out) <= 1.0 + 1e-9)


def test_fno_block_state_dict_roundtrip():
    b1  = FNOBlock(d_model=8, k_max=3, seed=0)
    sd  = b1.state_dict()
    b2  = FNOBlock(d_model=8, k_max=3, seed=99)
    b2.load_state_dict(sd)
    x   = np.ones((10, 8))
    assert np.allclose(b1.forward(x), b2.forward(x))


# ===========================================================================
# FNO1d
# ===========================================================================

def test_fno1d_output_shape_batch():
    fno = FNO1d(d_in=4, d_out=2, d_model=16, n_layers=2, k_max=4)
    x   = np.random.default_rng(0).standard_normal((32, 4))
    y   = fno.forward(x)
    assert y.shape == (32, 2)


def test_fno1d_output_shape_single():
    fno = FNO1d(d_in=4, d_out=4, d_model=8, n_layers=1, k_max=2)
    x   = np.ones(4)
    y   = fno.forward(x)
    assert y.shape == (4,)


def test_fno1d_resolution_invariance():
    """Same FNO, different n_points -- all should work."""
    fno = FNO1d(d_in=2, d_out=2, d_model=8, n_layers=1, k_max=3)
    for n in [4, 8, 16, 64, 128]:
        out = fno.forward(np.ones((n, 2)))
        assert out.shape == (n, 2), f"failed at n={n}"


def test_fno1d_param_count_positive():
    fno = FNO1d(d_in=4, d_out=4, d_model=16, n_layers=2, k_max=4)
    assert fno.param_count() > 0


def test_fno1d_save_load(tmp_path):
    fno  = FNO1d(d_in=4, d_out=2, d_model=8, n_layers=1, k_max=2, seed=7)
    path = str(tmp_path / "fno.npz")
    fno.save(path)
    fno2 = FNO1d.load(path)
    x    = np.random.default_rng(0).standard_normal((16, 4))
    assert np.allclose(fno.forward(x), fno2.forward(x))


def test_fno1d_fit_raises_without_torch():
    fno = FNO1d(d_in=2, d_out=2)
    X   = np.random.default_rng(0).standard_normal((10, 8, 2))
    Y   = np.random.default_rng(1).standard_normal((10, 8, 2))
    try:
        import torch
        pytest.skip("torch is installed; fit() path not tested here")
    except ImportError:
        with pytest.raises(ImportError, match="PyTorch"):
            fno.fit(X, Y)


def test_fno1d_repr():
    fno = FNO1d(d_in=4, d_out=2)
    s   = repr(fno)
    assert "FNO1d" in s
    assert "params" in s


# ===========================================================================
# FNOPredictiveRelay
# ===========================================================================

def test_fno_predictive_relay_output_shape():
    relay = FNOPredictiveRelay(n_dims=8, window=4)
    x_rec, stats = relay.relay(np.ones(8))
    assert x_rec.shape == (8,)


def test_fno_predictive_relay_stats_keys():
    relay = FNOPredictiveRelay(n_dims=8, window=4)
    _, stats = relay.relay(np.ones(8))
    for k in ("prediction_error", "reconstruction_error",
              "k_sent", "sparsity", "compression_ratio", "fno_trained"):
        assert k in stats


def test_fno_predictive_relay_threshold_sparsifies():
    relay = FNOPredictiveRelay(n_dims=16, window=4, threshold=1e10)
    _, stats = relay.relay(np.ones(16))
    assert stats["k_sent"] == 0   # threshold so high nothing is sent


def test_fno_predictive_relay_buffer_advances():
    relay = FNOPredictiveRelay(n_dims=4, window=3)
    for _ in range(5):
        relay.relay(np.random.default_rng(0).standard_normal(4))
    assert relay._t == 5


def test_fno_predictive_collect_training_data():
    relay    = FNOPredictiveRelay(n_dims=4, window=3)
    signals  = np.random.default_rng(0).standard_normal((20, 4))
    X, Y     = relay.collect_training_data(signals)
    assert X.shape[1] == 3   # window
    assert X.shape[2] == 4   # n_dims
    assert X.shape == Y.shape


def test_fno_predictive_relay_save_load(tmp_path):
    relay = FNOPredictiveRelay(n_dims=4, window=3, seed=0)
    path  = str(tmp_path / "pred_relay.npz")
    relay.save(path)
    relay2 = FNOPredictiveRelay.load_weights(path, n_dims=4, window=3, seed=0)
    assert relay2._trained


# ===========================================================================
# FNOOscillatoryRelay
# ===========================================================================

def test_fno_oscillatory_output_shape():
    relay = FNOOscillatoryRelay(n_dims=8)
    x_rec, stats = relay.transmit(np.ones(8))
    assert x_rec.shape == (8,)


def test_fno_oscillatory_stats_keys():
    relay = FNOOscillatoryRelay(n_dims=8)
    _, stats = relay.transmit(np.ones(8))
    for k in ("reconstruction_error", "mean_phase", "phase_spread", "fno_trained"):
        assert k in stats


def test_fno_oscillatory_phase_in_range():
    relay = FNOOscillatoryRelay(n_dims=8)
    for _ in range(10):
        relay.transmit(np.random.default_rng(0).standard_normal(8))
    assert np.all(relay._phase >= 0)
    assert np.all(relay._phase < 2 * np.pi)


def test_fno_oscillatory_encode_decode_roundtrip():
    relay = FNOOscillatoryRelay(n_dims=8, noise_std=0.0)
    x     = np.random.default_rng(0).uniform(-0.5, 0.5, 8)
    carrier, phase = relay.encode(x)
    x_rec          = relay.decode(carrier, phase)
    assert np.allclose(x, x_rec, atol=1e-6)


# ===========================================================================
# FNOHierarchicalRelay
# ===========================================================================

def test_fno_hierarchical_encode_shape():
    relay = FNOHierarchicalRelay(dims=[16, 8, 4])
    h     = relay.encode(np.ones(16))
    assert h.shape == (4,)


def test_fno_hierarchical_decode_shape():
    relay = FNOHierarchicalRelay(dims=[16, 8, 4])
    h     = relay.encode(np.ones(16))
    x_rec = relay.decode(h)
    assert x_rec.shape == (16,)


def test_fno_hierarchical_relay_stats():
    relay     = FNOHierarchicalRelay(dims=[16, 4])
    x_rec, s  = relay.relay(np.ones(16))
    assert x_rec.shape == (16,)
    assert "total_compression" in s
    assert s["total_compression"] == pytest.approx(4.0)


def test_fno_hierarchical_three_levels():
    relay = FNOHierarchicalRelay(dims=[32, 8, 2])
    x_rec, s = relay.relay(np.ones(32))
    assert x_rec.shape == (32,)
    assert s["total_compression"] == pytest.approx(16.0)


# ===========================================================================
# FNODecoherenceModel
# ===========================================================================

def test_fno_decoherence_output_shape():
    model = FNODecoherenceModel(capacity=8)
    c     = np.ones(8)
    c_new = model.step(c, dt=1.0)
    assert c_new.shape == (8,)


def test_fno_decoherence_untrained_decays():
    model = FNODecoherenceModel(capacity=4)
    model.calibrate_fallback(tau=5.0)
    c     = np.ones(4)
    c_new = model.step(c, dt=1.0)
    assert np.all(c_new < 1.0)
    assert np.all(c_new > 0.0)


def test_fno_decoherence_clipped_to_unit():
    model = FNODecoherenceModel(capacity=4)
    c     = np.ones(4) * 0.5
    c_new = model.step(c, dt=1.0)
    assert np.all(c_new >= 0.0)
    assert np.all(c_new <= 1.0)


def test_fno_decoherence_repr():
    model = FNODecoherenceModel(capacity=8)
    assert "FNODecoherenceModel" in repr(model)
