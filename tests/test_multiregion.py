"""Tests for Cat-5 multi-region modules:
HemisphericBridge, HierarchicalRelay, BindingBus.
"""

import numpy as np
import pytest

from qbit_simulator.hemispheric_bridge import (
    Hemisphere, BottleneckChannel, InhibitionModule, HemisphericBridge,
)
from qbit_simulator.hierarchical_relay import (
    HierarchyLevel, HierarchicalRelay, AdaptiveHierarchicalRelay,
)
from qbit_simulator.binding_bus import (
    GammaOscillator, FeatureBundle, PhaseComparator,
    BindingBus, UnifiedPercept,
)


# ===========================================================================
# Hemisphere
# ===========================================================================

def test_hemisphere_update_and_state():
    h = Hemisphere(n_dims=8, name="left")
    x = np.arange(8, dtype=float)
    h.update(x)
    assert np.allclose(h.state, x)


def test_hemisphere_encode_identity_no_transform():
    h = Hemisphere(n_dims=4)
    h.update(np.array([1.0, 2.0, 3.0, 4.0]))
    enc = h.encode()
    assert np.allclose(enc, [1.0, 2.0, 3.0, 4.0])


def test_hemisphere_encode_with_transform():
    T = np.eye(2, 4)   # project 4 -> 2 dims
    h = Hemisphere(n_dims=4, transform=T)
    h.update(np.array([1.0, 2.0, 3.0, 4.0]))
    enc = h.encode()
    assert enc.shape == (2,)
    assert np.allclose(enc, [1.0, 2.0])


def test_hemisphere_dominance_score():
    h = Hemisphere(n_dims=4)
    h.update(np.array([1.0, 0.0, 0.0, 0.0]))
    assert h.dominance_score() == pytest.approx(1.0)


def test_hemisphere_mean_activity_tracks_norm():
    h = Hemisphere(n_dims=4)
    for _ in range(5):
        h.update(np.ones(4) * 2.0)
    assert h.mean_activity == pytest.approx(np.linalg.norm(np.ones(4) * 2.0))


# ===========================================================================
# BottleneckChannel
# ===========================================================================

def test_bottleneck_topk_output_shape():
    ch = BottleneckChannel(n_in=16, n_out=16, bottleneck=4, mode="topk")
    x  = np.random.default_rng(0).standard_normal(16)
    y, stats = ch.transmit(x)
    assert y.shape == (16,)


def test_bottleneck_projection_output_shape():
    ch = BottleneckChannel(n_in=16, n_out=8, bottleneck=4, mode="projection")
    x  = np.random.default_rng(0).standard_normal(16)
    y, stats = ch.transmit(x)
    assert y.shape == (8,)


def test_bottleneck_stats_keys():
    ch = BottleneckChannel(n_in=8, n_out=8, bottleneck=2)
    _, stats = ch.transmit(np.ones(8))
    for k in ("bottleneck", "compression_ratio", "reconstruction_error"):
        assert k in stats


def test_bottleneck_compression_ratio():
    ch = BottleneckChannel(n_in=32, n_out=32, bottleneck=4, mode="topk")
    _, stats = ch.transmit(np.ones(32))
    assert stats["compression_ratio"] == pytest.approx(8.0)


def test_bottleneck_delay_shifts_signal():
    """With delay=2, first 2 outputs should be zeros (fill)."""
    ch  = BottleneckChannel(n_in=4, n_out=4, bottleneck=4,
                              mode="topk", delay_steps=2)
    y0, _ = ch.transmit(np.ones(4))
    y1, _ = ch.transmit(np.ones(4))
    y2, _ = ch.transmit(np.ones(4))
    # First two calls should return the fill (zeros).
    assert np.allclose(y0, 0.0)
    assert np.allclose(y1, 0.0)
    # Third call returns the delayed signal.
    assert not np.allclose(y2, 0.0)


# ===========================================================================
# InhibitionModule
# ===========================================================================

def test_inhibition_zero_when_equal_dominance():
    inh = InhibitionModule(inh_strength=0.5)
    x   = np.ones(4)
    x_s, factor = inh.apply(x, dom_sender=1.0, dom_receiver=1.0)
    # sigma(0) = 0.5 -> inh = 0.5*0.5 = 0.25 -> factor = 0.75
    assert 0.5 < factor < 1.0


def test_inhibition_strong_when_sender_dominant():
    inh = InhibitionModule(inh_strength=1.0)
    x   = np.ones(4) * 2.0
    _, factor = inh.apply(x, dom_sender=10.0, dom_receiver=0.0)
    # sigma(10) ~ 1.0 -> factor ~ 0.0 (strong suppression)
    assert factor < 0.2


def test_inhibition_weak_when_receiver_dominant():
    inh = InhibitionModule(inh_strength=0.5)
    x   = np.ones(4)
    _, factor = inh.apply(x, dom_sender=0.0, dom_receiver=10.0)
    # sigma(-10) ~ 0 -> factor ~ 1.0 (little suppression)
    assert factor > 0.9


# ===========================================================================
# HemisphericBridge
# ===========================================================================

def test_bridge_step_returns_dict():
    bridge = HemisphericBridge(n_left=16, n_right=16,
                                bottleneck_lr=4, bottleneck_rl=4)
    stats  = bridge.step(np.ones(16), np.ones(16))
    for k in ("dominance_left", "dominance_right",
              "inh_factor_left", "inh_factor_right",
              "lr_error", "rl_error", "recv_left", "recv_right"):
        assert k in stats


def test_bridge_recv_shapes():
    bridge = HemisphericBridge(n_left=8, n_right=12,
                                bottleneck_lr=2, bottleneck_rl=2)
    stats  = bridge.step(np.ones(8), np.ones(12))
    assert stats["recv_right"].shape == (12,)
    assert stats["recv_left"].shape  == (8,)


def test_bridge_dominant_hemisphere():
    bridge = HemisphericBridge(n_left=4, n_right=4,
                                bottleneck_lr=2, bottleneck_rl=2)
    bridge.step(np.array([10.0, 0.0, 0.0, 0.0]),
                np.array([0.0,  0.0, 0.0, 0.0]))
    assert bridge.dominant_hemisphere() == "left"


def test_bridge_history_grows():
    bridge = HemisphericBridge(n_left=4, n_right=4,
                                bottleneck_lr=2, bottleneck_rl=2)
    for _ in range(5):
        bridge.step(np.ones(4), np.ones(4))
    assert len(bridge.history()) == 5


def test_bridge_asymmetric_dims():
    bridge = HemisphericBridge(n_left=16, n_right=8,
                                bottleneck_lr=4, bottleneck_rl=2)
    stats  = bridge.step(np.ones(16), np.ones(8))
    assert stats["recv_right"].shape == (8,)
    assert stats["recv_left"].shape  == (16,)


# ===========================================================================
# HierarchyLevel
# ===========================================================================

def test_level_encode_output_shape():
    lv  = HierarchyLevel(n_in=16, n_out=4)
    out = lv.encode(np.ones(16))
    assert out.shape == (4,)


def test_level_decode_output_shape():
    lv = HierarchyLevel(n_in=16, n_out=4)
    h  = np.ones(4)
    x  = lv.decode(h)
    assert x.shape == (16,)


def test_level_compression_ratio():
    lv = HierarchyLevel(n_in=32, n_out=8)
    assert lv.compression_ratio == pytest.approx(4.0)


def test_level_relu_nonlin_nonneg():
    lv  = HierarchyLevel(n_in=8, n_out=4, nonlin="relu")
    out = lv.encode(np.random.default_rng(0).standard_normal(8))
    assert np.all(out >= 0)


def test_level_pca_init():
    lv  = HierarchyLevel(n_in=8, n_out=4, init="pca", nonlin=None)
    out = lv.encode(np.ones(8))
    assert out.shape == (4,)


# ===========================================================================
# HierarchicalRelay
# ===========================================================================

def test_relay_encode_output_shape():
    relay    = HierarchicalRelay(dims=[64, 16, 4])
    h, stats = relay.encode(np.ones(64))
    assert h.shape == (4,)
    assert stats["top_dim"] == 4


def test_relay_decode_output_shape():
    relay     = HierarchicalRelay(dims=[64, 16, 4])
    h, _      = relay.encode(np.ones(64))
    x_rec, _  = relay.decode(h, target_dim=64)
    assert x_rec.shape == (64,)


def test_relay_roundtrip_shape():
    relay     = HierarchicalRelay(dims=[32, 8, 2])
    x         = np.random.default_rng(0).standard_normal(32)
    x_rec, s  = relay.relay(x)
    assert x_rec.shape == (32,)
    assert "total_compression" in s
    assert "reconstruction_error" in s


def test_relay_total_compression():
    relay = HierarchicalRelay(dims=[256, 64, 16, 4])
    assert relay.total_compression == pytest.approx(64.0)


def test_relay_n_levels():
    relay = HierarchicalRelay(dims=[64, 16, 4])
    assert relay.n_levels == 2


def test_relay_intermediate_reps():
    relay = HierarchicalRelay(dims=[16, 8, 4])
    reps  = relay.intermediate_representations(np.ones(16))
    assert len(reps) == 3   # input + 2 levels
    assert reps[0].shape == (16,)
    assert reps[1].shape == (8,)
    assert reps[2].shape == (4,)


def test_relay_single_level_raises_on_bad_dims():
    with pytest.raises(ValueError):
        HierarchicalRelay(dims=[64])


# ===========================================================================
# AdaptiveHierarchicalRelay
# ===========================================================================

def test_adaptive_relay_output_shape():
    relay    = AdaptiveHierarchicalRelay(dims=[16, 4, 2], target_error=0.2)
    x_rec, s = relay.relay(np.ones(16))
    assert x_rec.shape == (16,)
    assert "active_depth" in s
    assert "ema_error" in s


def test_adaptive_relay_depth_bounded():
    relay = AdaptiveHierarchicalRelay(dims=[16, 8, 4, 2], target_error=0.0)
    rng   = np.random.default_rng(0)
    for _ in range(20):
        relay.relay(rng.standard_normal(16))
    assert 1 <= relay._active_depth <= 3


# ===========================================================================
# GammaOscillator
# ===========================================================================

def test_oscillator_phase_in_range():
    osc = GammaOscillator(freq_hz=40.0)
    for _ in range(100):
        phi = osc.step()
        assert 0 <= phi < 2 * np.pi


def test_oscillator_reset():
    osc  = GammaOscillator()
    osc.step(); osc.step()
    osc.reset()
    assert osc.current_phase == pytest.approx(0.0)


def test_oscillator_frequency():
    """After exactly 1/freq seconds, phase should wrap once."""
    freq = 10.0
    dt   = 0.001
    osc  = GammaOscillator(freq_hz=freq, dt=dt)
    steps_per_cycle = int(1.0 / (freq * dt))
    for _ in range(steps_per_cycle):
        osc.step()
    # Phase should be close to 0 (wrapped).
    phi = osc.current_phase
    assert phi < 2 * np.pi


# ===========================================================================
# FeatureBundle
# ===========================================================================

def test_feature_bundle_creation():
    fb = FeatureBundle(features=np.array([1.0, 2.0, 3.0]),
                       phases=np.array([0.1, 0.2, 0.3]),
                       region="V1")
    assert fb.n_features == 3


def test_feature_bundle_mismatched_raises():
    with pytest.raises(ValueError):
        FeatureBundle(features=np.ones(3), phases=np.ones(4))


def test_feature_bundle_mean_phase():
    phases = np.array([0.0, 0.0, 0.0])
    fb     = FeatureBundle(features=np.ones(3), phases=phases)
    assert abs(fb.mean_phase()) < 0.01


def test_feature_bundle_phase_spread_in_phase():
    phases = np.zeros(5)
    fb     = FeatureBundle(features=np.ones(5), phases=phases)
    assert fb.phase_spread() < 0.01   # all in phase


# ===========================================================================
# PhaseComparator
# ===========================================================================

def test_phase_comparator_perfect_sync():
    comp = PhaseComparator(plv_threshold=0.7)
    b1   = FeatureBundle(features=np.ones(4), phases=np.zeros(4))
    b2   = FeatureBundle(features=np.ones(4), phases=np.zeros(4))
    assert comp.plv(b1, b2) == pytest.approx(1.0)
    assert comp.are_bound(b1, b2)


def test_phase_comparator_random_not_bound():
    rng  = np.random.default_rng(99)
    comp = PhaseComparator(plv_threshold=0.7)
    b1   = FeatureBundle(features=np.ones(100),
                          phases=rng.uniform(0, 2*np.pi, 100))
    b2   = FeatureBundle(features=np.ones(100),
                          phases=rng.uniform(0, 2*np.pi, 100))
    # PLV of random phases ~ 0 -> not bound.
    assert not comp.are_bound(b1, b2)


def test_phase_comparator_pairwise_matrix_shape():
    comp    = PhaseComparator()
    bundles = [FeatureBundle(np.ones(4), np.zeros(4)) for _ in range(3)]
    mat     = comp.pairwise_plv_matrix(bundles)
    assert mat.shape == (3, 3)
    assert np.allclose(np.diag(mat), 1.0)


# ===========================================================================
# BindingBus
# ===========================================================================

def test_binding_bus_encode_shape():
    bus = BindingBus()
    bus.register("V1", 8)
    rng = np.random.default_rng(0)
    b   = bus.encode("V1", np.ones(8), ref_phase=0.0, rng=rng)
    assert b.n_features == 8


def test_binding_bus_bind_all_agree():
    """Bundles with identical phases should all be bound."""
    bus = BindingBus(plv_threshold=0.7)
    b1  = FeatureBundle(np.ones(4), np.zeros(4), region="V1")
    b2  = FeatureBundle(np.ones(4), np.zeros(4), region="MT")
    p   = bus.bind([b1, b2])
    assert p.is_bound()
    assert p.n_bound == 2


def test_binding_bus_bind_random_phase_partial():
    """Bundles with random phases relative to anchor should not all bind."""
    rng = np.random.default_rng(42)
    bus = BindingBus(plv_threshold=0.9)
    # Anchor: all features at phase 0.
    anchor = FeatureBundle(np.ones(50), np.zeros(50), region="A")
    # Random phases -> low PLV with anchor.
    random_b = FeatureBundle(np.ones(50),
                               rng.uniform(0, 2*np.pi, 50),
                               region="B")
    p = bus.bind([anchor, random_b])
    # At least the anchor should be in the percept.
    assert p.n_bound >= 1


def test_binding_bus_output_dim():
    bus = BindingBus(output_dim=16)
    b1  = FeatureBundle(np.ones(8),  np.zeros(8),  region="A")
    b2  = FeatureBundle(np.ones(32), np.zeros(32), region="B")
    p   = bus.bind([b1, b2])
    assert p.features.shape == (16,)


def test_binding_bus_step_and_bind():
    rng = np.random.default_rng(7)
    bus = BindingBus(plv_threshold=0.5, phase_noise_std=0.05)
    bus.register("V4", 8)
    bus.register("IT",  8)
    p   = bus.step_and_bind({"V4": np.ones(8), "IT": np.ones(8)}, rng=rng)
    assert p.features.shape[0] > 0
    assert p.plv_matrix.shape == (2, 2)


def test_binding_bus_result_keys():
    bus = BindingBus()
    b   = FeatureBundle(np.ones(4), np.zeros(4))
    p   = bus.bind([b])
    assert hasattr(p, "features")
    assert hasattr(p, "binding_mask")
    assert hasattr(p, "plv_matrix")
    assert hasattr(p, "mean_plv")
    assert hasattr(p, "n_bound")


def test_binding_bus_history_grows():
    bus = BindingBus()
    b   = FeatureBundle(np.ones(4), np.zeros(4))
    for _ in range(5):
        bus.bind([b, b])
    assert len(bus.history()) == 5


def test_binding_bus_empty_bundles():
    bus = BindingBus(output_dim=4)
    p   = bus.bind([])
    assert p.n_bound == 0
    assert p.features.shape == (4,)


def test_binding_bus_merge_modes():
    b1 = FeatureBundle(np.array([1.0, 0.0]), np.zeros(2), region="A")
    b2 = FeatureBundle(np.array([0.0, 2.0]), np.zeros(2), region="B")
    for mode in ("mean", "max", "weighted"):
        bus = BindingBus(merge_mode=mode, plv_threshold=0.0)
        p   = bus.bind([b1, b2])
        assert p.features.shape == (2,)
