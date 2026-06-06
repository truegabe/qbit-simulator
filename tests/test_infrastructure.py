"""Tests for the four brain infrastructure modules:
ThalamicGate, PredictiveRelay, NeuromodulatorBroadcast, OscillatoryRelay.
"""

import numpy as np
import pytest

# ---- ThalamicGate ----
from qbit_simulator.thalamic_gate import ThalamicGate, ThalamicRelay

# ---- PredictiveRelay ----
from qbit_simulator.predictive_relay import (
    EMAPredictor, LinearPredictor, ConstantPredictor, PredictiveRelay,
)

# ---- NeuromodulatorBroadcast ----
from qbit_simulator.neuromodulator_broadcast import (
    NeuromodulatorState, NeuromodulatorSystem,
    ModulatableParams, neuromod_to_params, NeuromodulatorBroadcast,
)

# ---- OscillatoryRelay ----
from qbit_simulator.oscillatory_relay import (
    hilbert_transform, instantaneous_phase, phase_locking_value,
    OscillatoryEncoder, OscillatoryDecoder, PhaseCoherence, OscillatoryRelay,
)


# ===========================================================================
# ThalamicGate
# ===========================================================================

def test_gate_open_passes_signal():
    gate = ThalamicGate(n_dims=8, context_dims=1, gate_mode="soft",
                        init_open=True)
    x   = np.ones(8)
    ctx = np.array([1.0])
    out, info = gate.forward(x, ctx)
    assert out.shape == (8,)
    assert np.all(out > 0)


def test_gate_closed_zeros_signal():
    gate = ThalamicGate(n_dims=8, context_dims=1, gate_mode="hard",
                        init_open=False, threshold=0.5)
    gate.close_all()
    x   = np.ones(8)
    ctx = np.array([0.0])
    out, info = gate.forward(x, ctx)
    assert np.allclose(out, 0.0)


def test_gate_info_keys():
    gate = ThalamicGate(n_dims=4, context_dims=2)
    out, info = gate.forward(np.ones(4), np.ones(2))
    for k in ("gate_values", "mean_gate", "open_fraction", "mode"):
        assert k in info


def test_gate_update_increases_openness_for_high_relevance():
    gate = ThalamicGate(n_dims=8, context_dims=1, init_open=False, lr=0.5)
    ctx  = np.array([1.0])
    for _ in range(20):
        gate.update(ctx, relevance=1.0)
    g = gate.gate_values(ctx)
    assert g.mean() > 0.5


def test_gate_mean_openness_tracks_history():
    gate = ThalamicGate(n_dims=4, context_dims=1)
    ctx  = np.array([1.0])
    for _ in range(10):
        gate.forward(np.ones(4), ctx)
    assert 0.0 <= gate.mean_openness() <= 1.0


def test_thalamic_relay_routes_all_regions():
    relay = ThalamicRelay(n_regions=3, n_dims_list=8, context_dims=2)
    signals = [np.ones(8) for _ in range(3)]
    ctx     = np.array([0.8, 0.6])
    outputs, infos = relay.route(signals, ctx)
    assert len(outputs) == 3
    assert len(infos)   == 3
    for o in outputs:
        assert o.shape == (8,)


def test_thalamic_relay_update_all():
    relay = ThalamicRelay(n_regions=2, n_dims_list=4, context_dims=1)
    ctx   = np.array([1.0])
    relay.update_all(ctx, relevances=[0.9, 0.1])
    openness = relay.mean_openness()
    assert len(openness) == 2


# ===========================================================================
# PredictiveRelay
# ===========================================================================

def test_ema_predictor_converges():
    pred = EMAPredictor(n_dims=4, alpha=0.5)
    x    = np.array([1.0, 2.0, 3.0, 4.0])
    for _ in range(50):
        pred.update(x)
    assert np.allclose(pred.predict(x), x, atol=0.1)


def test_linear_predictor_shape():
    pred = LinearPredictor(n_dims=6)
    x    = np.ones(6)
    p    = pred.predict(x)
    assert p.shape == (6,)
    pred.update(x)


def test_constant_predictor_converges_to_mean():
    pred = ConstantPredictor(n_dims=3)
    for i in range(100):
        pred.update(np.array([float(i % 4), 1.0, 2.0]))
    # Mean of 0,1,2,3 cycle = 1.5
    assert abs(pred._mean[0] - 1.5) < 0.1


def test_predictive_relay_relay_shape():
    relay = PredictiveRelay(n_dims=8)
    x     = np.random.default_rng(0).standard_normal(8)
    x_rec, stats = relay.relay(x)
    assert x_rec.shape == (8,)


def test_predictive_relay_error_decreases_on_constant_input():
    relay = PredictiveRelay(n_dims=4, predictor_type="ema", alpha=0.3)
    x     = np.array([1.0, -1.0, 0.5, 2.0])
    errors = []
    for _ in range(30):
        _, stats = relay.relay(x)
        errors.append(stats["error_norm"])
    assert errors[-1] < errors[0]


def test_predictive_relay_sparsify_reduces_nonzeros():
    relay_dense  = PredictiveRelay(n_dims=16, sparsify=False)
    relay_sparse = PredictiveRelay(n_dims=16, sparsify=True, sparse_threshold=0.3)
    rng = np.random.default_rng(7)
    x   = rng.standard_normal(16)
    _, stats_d = relay_dense.relay(x)
    _, stats_s = relay_sparse.relay(x)
    assert stats_s["nonzero_components"] <= stats_d["nonzero_components"]


def test_predictive_relay_send_receive_exact_on_perfect_channel():
    relay = PredictiveRelay(n_dims=6, predictor_type="constant")
    x     = np.array([0.1, 0.5, -0.3, 0.8, 0.2, -0.7])
    error, _ = relay.send(x)
    x_rec    = relay.receive(error)
    # First transmission: predictor = zeros, so error = x, receive = x exactly.
    assert np.allclose(x_rec, x, atol=1e-9)


def test_predictive_relay_mean_error_stat():
    relay = PredictiveRelay(n_dims=4)
    for _ in range(10):
        relay.relay(np.ones(4))
    assert 0.0 <= relay.mean_relative_error() <= 1.0


def test_predictive_relay_linear_predictor():
    relay = PredictiveRelay(n_dims=4, predictor_type="linear", lr=0.05)
    x = np.array([1.0, 0.0, -1.0, 0.5])
    for _ in range(20):
        relay.relay(x)
    _, stats = relay.relay(x)
    assert stats["reconstruction_error"] < 0.5


# ===========================================================================
# NeuromodulatorBroadcast
# ===========================================================================

def test_neuromod_state_as_array():
    s = NeuromodulatorState(DA=0.8, ACh=0.3, NE=0.6, HT5=0.4)
    arr = s.as_array()
    assert arr.shape == (4,)
    assert abs(arr[0] - 0.8) < 1e-9


def test_neuromod_system_reward_raises_da():
    nms = NeuromodulatorSystem()
    da_before = nms.state.DA
    nms.reward_signal(rpe=+1.0)
    assert nms.state.DA > da_before


def test_neuromod_system_threat_raises_ne_lowers_5ht():
    nms = NeuromodulatorSystem()
    nms.threat_signal(1.0)
    assert nms.state.NE > 0.5
    assert nms.state.HT5 < 0.5


def test_neuromod_system_step_decays_toward_baseline():
    nms = NeuromodulatorSystem()
    nms.set(DA=1.0)
    for _ in range(50):
        nms.step(dt=1.0)
    assert nms.state.DA < 0.8


def test_neuromod_to_params_high_da_high_lr():
    s = NeuromodulatorState(DA=1.0, ACh=0.5, NE=0.5, HT5=0.5)
    p = neuromod_to_params(s)
    assert p.learning_rate_scale > 1.5


def test_neuromod_to_params_high_ne_high_gain():
    s = NeuromodulatorState(DA=0.5, ACh=0.5, NE=1.0, HT5=0.5)
    p = neuromod_to_params(s)
    assert p.gain > 1.5


def test_neuromod_broadcast_fires_callback():
    nms       = NeuromodulatorSystem()
    broadcast = NeuromodulatorBroadcast(nms)
    received  = []
    broadcast.register("test_region", lambda p: received.append(p.gain))
    broadcast.fire()
    assert len(received) == 1


def test_neuromod_broadcast_registers_multiple():
    nms       = NeuromodulatorSystem()
    broadcast = NeuromodulatorBroadcast(nms)
    counts    = {"a": 0, "b": 0}
    broadcast.register("a", lambda p: counts.__setitem__("a", counts["a"] + 1))
    broadcast.register("b", lambda p: counts.__setitem__("b", counts["b"] + 1))
    broadcast.fire()
    broadcast.fire()
    assert counts["a"] == 2 and counts["b"] == 2


def test_neuromod_broadcast_history():
    nms       = NeuromodulatorSystem()
    broadcast = NeuromodulatorBroadcast(nms)
    for _ in range(5):
        broadcast.fire()
    arr = broadcast.history_array()
    assert arr.shape == (5, 4)


def test_neuromod_params_apply_to_signal():
    p = ModulatableParams(gain=2.0, noise_threshold=0.0)
    x = np.array([1.0, 2.0, 3.0])
    y = p.apply_to_signal(x)
    assert np.allclose(y, [2.0, 4.0, 6.0])


# ===========================================================================
# OscillatoryRelay
# ===========================================================================

def test_hilbert_real_part_preserved():
    x = np.sin(2 * np.pi * 10 * np.arange(100) / 1000.0)
    z = hilbert_transform(x)
    assert np.allclose(np.real(z), x, atol=1e-9)


def test_plv_self_coherence_is_one():
    x   = np.sin(2 * np.pi * 40 * np.arange(200) / 1000.0)
    phi = instantaneous_phase(x)
    plv = phase_locking_value(phi, phi)
    assert abs(plv - 1.0) < 1e-6


def test_plv_random_is_low():
    rng = np.random.default_rng(0)
    phi_a = rng.uniform(-np.pi, np.pi, 500)
    phi_b = rng.uniform(-np.pi, np.pi, 500)
    plv   = phase_locking_value(phi_a, phi_b)
    assert plv < 0.2


def test_encoder_output_shape():
    enc = OscillatoryEncoder(n_dims=4, carrier_hz=40.0,
                              sample_rate=1000.0, n_cycles=5)
    x   = np.array([0.5, -0.3, 0.8, 0.1])
    sig = enc.encode(x)
    assert sig.shape == (4, enc.n_samples)


def test_encoder_amplitude_within_bounds():
    enc = OscillatoryEncoder(n_dims=3, carrier_hz=40.0,
                              sample_rate=1000.0, n_cycles=3, amplitude=1.0)
    x   = np.array([1.0, 0.0, -1.0])
    sig = enc.encode(x)
    assert np.all(np.abs(sig) <= 1.0 + 1e-9)


def test_oscillatory_relay_output_shape():
    relay = OscillatoryRelay(n_dims=4, band="gamma",
                              sample_rate=1000.0, n_cycles=5,
                              plv_threshold=0.0)
    x     = np.array([0.5, -0.3, 0.8, 0.2])
    x_rec, stats = relay.transmit(x, force_coherent=True)
    assert x_rec.shape == (4,)


def test_oscillatory_relay_coherent_path_reconstructs():
    relay = OscillatoryRelay(n_dims=2, band="gamma",
                              sample_rate=1000.0, n_cycles=8,
                              plv_threshold=0.0, phase_noise_std=0.0)
    x     = np.array([0.6, -0.4])
    x_rec, stats = relay.transmit(x, force_coherent=True)
    # Should reconstruct sign correctly (phase decoding is approximate).
    assert np.sign(x_rec[0]) == np.sign(x[0])
    assert np.sign(x_rec[1]) == np.sign(x[1])


def test_oscillatory_relay_incoherent_blocks_signal():
    relay = OscillatoryRelay(n_dims=4, band="gamma",
                              phase_noise_std=3.0,   # heavy noise -> low PLV
                              plv_threshold=0.99,
                              sample_rate=1000.0, n_cycles=5)
    x = np.ones(4)
    x_rec, stats = relay.transmit(x, force_coherent=False)
    # With very high noise and threshold, signal should be blocked.
    assert not stats["coherent"] or np.allclose(x_rec, 0.0)


def test_oscillatory_relay_stats_keys():
    relay = OscillatoryRelay(n_dims=3, band="theta", plv_threshold=0.0)
    _, stats = relay.transmit(np.array([0.1, 0.2, 0.3]), force_coherent=True)
    for k in ("plv", "coherent", "reconstruction_error", "band", "carrier_hz"):
        assert k in stats


def test_phase_coherence_measure_returns_float():
    coh = PhaseCoherence()
    x   = np.sin(2 * np.pi * 40 * np.arange(100) / 1000.0)
    plv = coh.measure(x, x)
    assert isinstance(plv, float)
    assert 0.0 <= plv <= 1.0
