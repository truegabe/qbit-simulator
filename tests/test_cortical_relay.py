"""Tests for CorticalRelay -- compressed qudit communication bus."""

import numpy as np
import pytest

from qbit_simulator.cortical_relay import (
    QuditCodebook, QuditChannel, CorticalRelay, RelayProfiler, relay_connect,
)


# ---------------------------------------------------------------------------
# QuditCodebook
# ---------------------------------------------------------------------------

def test_codebook_encode_shape():
    cb = QuditCodebook(n_dims=16, n_symbols=4, d=10)
    x = np.random.default_rng(0).standard_normal(16)
    syms = cb.encode(x)
    assert syms.shape == (4,)
    assert syms.dtype in (np.int32, np.int64)
    assert all(0 <= s < 10 for s in syms)


def test_codebook_decode_shape():
    cb = QuditCodebook(n_dims=16, n_symbols=4, d=10)
    syms = np.array([3, 7, 1, 9], dtype=np.int32)
    x_hat = cb.decode(syms)
    assert x_hat.shape == (16,)


def test_codebook_roundtrip_improves_with_more_symbols():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(32)
    e_coarse = QuditCodebook(n_dims=32, n_symbols=2,  d=10, rng=rng).reconstruction_error(x)
    e_fine   = QuditCodebook(n_dims=32, n_symbols=16, d=10, rng=rng).reconstruction_error(x)
    assert e_fine < e_coarse


def test_codebook_fit_reduces_error():
    rng = np.random.default_rng(7)
    data = rng.standard_normal((200, 32))
    cb_default = QuditCodebook(n_dims=32, n_symbols=8, d=10, rng=rng)
    cb_fitted  = QuditCodebook(n_dims=32, n_symbols=8, d=10, rng=rng)
    cb_fitted.fit(data)
    x = data[0]
    assert cb_fitted.reconstruction_error(x) <= cb_default.reconstruction_error(x) + 0.5


def test_codebook_symbols_in_range():
    rng = np.random.default_rng(0)
    cb = QuditCodebook(n_dims=8, n_symbols=5, d=10, rng=rng)
    for _ in range(50):
        x = rng.standard_normal(8)
        s = cb.encode(x)
        assert all(0 <= v < 10 for v in s)


def test_compression_ratio_positive():
    cb = QuditCodebook(n_dims=64, n_symbols=4, d=10)
    assert cb.compression_ratio() > 1.0


# ---------------------------------------------------------------------------
# QuditChannel
# ---------------------------------------------------------------------------

def test_channel_perfect_transmit():
    ch = QuditChannel(d=10, error_rate=0.0)
    syms = np.array([3, 1, 4, 1, 5], dtype=np.int32)
    recv, latency = ch.transmit(syms)
    assert np.array_equal(recv, syms)
    assert latency == pytest.approx(5.0)


def test_channel_noisy_flips_some():
    rng = np.random.default_rng(99)
    ch = QuditChannel(d=10, error_rate=0.5, rng=rng)
    syms = np.zeros(100, dtype=np.int32)
    recv, _ = ch.transmit(syms)
    # With p=0.5, roughly half should be flipped away from 0.
    n_flipped = int((recv != 0).sum())
    assert 20 < n_flipped < 80


def test_channel_speedup_greater_than_one_when_symbols_less_than_dims():
    ch = QuditChannel(d=10, symbol_delay=1.0, spike_delay=1.0)
    # 4 symbols vs 48 raw bits -> 12x speedup.
    assert ch.speedup(n_dims=48, n_symbols=4) == pytest.approx(12.0)


def test_channel_latency_scales_with_symbols():
    ch = QuditChannel(symbol_delay=2.0, error_rate=0.0)
    _, lat = ch.transmit(np.zeros(6, dtype=np.int32))
    assert lat == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# CorticalRelay
# ---------------------------------------------------------------------------

def test_relay_output_shape():
    relay = CorticalRelay(n_dims=16, n_symbols=4, d=10)
    x = np.ones(16)
    x_hat, stats = relay.transmit(x)
    assert x_hat.shape == (16,)


def test_relay_stats_keys():
    relay = CorticalRelay(n_dims=8, n_symbols=3, d=10)
    _, stats = relay.transmit(np.zeros(8))
    for key in ("speedup", "compression_ratio", "reconstruction_error",
                "latency_ms", "symbols_sent"):
        assert key in stats


def test_relay_speedup_positive():
    relay = CorticalRelay(n_dims=64, n_symbols=4, d=10,
                          symbol_delay=1.0, spike_delay=1.0)
    _, stats = relay.transmit(np.random.standard_normal(64))
    assert stats["speedup"] > 1.0


def test_relay_fit_then_transmit():
    rng = np.random.default_rng(3)
    data = rng.standard_normal((100, 32))
    relay = CorticalRelay(n_dims=32, n_symbols=8, d=10, rng=rng)
    relay.fit(data)
    x_hat, stats = relay.transmit(data[0])
    assert x_hat.shape == (32,)
    assert stats["reconstruction_error"] < 1.0


def test_relay_encode_decode_consistency():
    relay = CorticalRelay(n_dims=12, n_symbols=4, d=10)
    x = np.array([0.1, 0.5, -0.3, 0.8, 0.2, -0.1, 0.9, 0.0,
                  0.4, -0.7, 0.3, 0.6])
    syms = relay.encode_only(x)
    x_hat = relay.decode_only(syms)
    assert x_hat.shape == (12,)


def test_relay_perfect_channel_deterministic():
    relay = CorticalRelay(n_dims=8, n_symbols=3, d=10, error_rate=0.0)
    x = np.array([1.0, -1.0, 0.5, 0.0, 2.0, -0.5, 1.5, 0.3])
    x_hat1, _ = relay.transmit(x)
    x_hat2, _ = relay.transmit(x)
    assert np.allclose(x_hat1, x_hat2)


# ---------------------------------------------------------------------------
# RelayProfiler
# ---------------------------------------------------------------------------

def test_profiler_report_fields():
    relay = CorticalRelay(n_dims=16, n_symbols=4, d=10)
    prof = RelayProfiler(relay)
    rng = np.random.default_rng(0)
    for _ in range(20):
        prof.step(rng.standard_normal(16))
    report = prof.report()
    assert report["n_transmissions"] == 20
    assert "mean_reconstruction_error" in report
    assert "mean_speedup" in report
    assert report["mean_speedup"] > 1.0


def test_profiler_reset():
    relay = CorticalRelay(n_dims=8, n_symbols=2, d=10)
    prof = RelayProfiler(relay)
    prof.step(np.zeros(8))
    prof.reset()
    assert prof.report() == {}


# ---------------------------------------------------------------------------
# relay_connect  (end-to-end pipeline)
# ---------------------------------------------------------------------------

def test_relay_connect_pipeline():
    rng = np.random.default_rng(5)
    relay = CorticalRelay(n_dims=8, n_symbols=4, d=10, rng=rng)

    # Region A: simple linear transform (like a sensory encoder).
    W_a = rng.standard_normal((8, 4))
    region_a = lambda x: np.tanh(W_a @ x)

    # Region B: another linear transform (like a cognitive decoder).
    W_b = rng.standard_normal((4, 8))
    region_b = lambda x: np.tanh(W_b @ x)

    pipeline = relay_connect(region_a, region_b, relay)
    x_in = rng.standard_normal(4)
    out, stats = pipeline(x_in)
    assert out.shape == (4,)
    assert stats["speedup"] > 1.0
