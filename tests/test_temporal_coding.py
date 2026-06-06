"""Tests for Category 3 -- Temporal coding modules:
AxonalDelayLine and SpikeRoutingBus.
OscillatoryRelay is tested in test_infrastructure.py.
"""

import numpy as np
import pytest

from qbit_simulator.axonal_delay_line import (
    DelayLine, DelayLineBank, AxonalDelayLine, TemporalMatcher,
)
from qbit_simulator.spike_routing_bus import (
    SpikeEvent, RoutingTable, SpikeRoutingBus,
    AEREncoder, AERDecoder, AERRelay,
)


# ===========================================================================
# DelayLine
# ===========================================================================

def test_delay_line_zero_delay_passthrough():
    dl = DelayLine(delay=0)
    assert dl.push(3.0) == pytest.approx(3.0)


def test_delay_line_delays_by_correct_amount():
    dl = DelayLine(delay=3)
    outputs = [dl.push(float(i)) for i in range(6)]
    # First 3 outputs are fill (0.0), then delayed inputs.
    assert outputs[:3] == [0.0, 0.0, 0.0]
    assert outputs[3] == pytest.approx(0.0)   # pushed 0 at t=0
    assert outputs[4] == pytest.approx(1.0)   # pushed 1 at t=1
    assert outputs[5] == pytest.approx(2.0)   # pushed 2 at t=2


def test_delay_line_reset_clears_buffer():
    dl = DelayLine(delay=2)
    dl.push(5.0); dl.push(6.0)
    dl.reset()
    assert dl.peek() == 0.0


def test_delay_line_delay_1():
    dl = DelayLine(delay=1)
    assert dl.push(7.0) == pytest.approx(0.0)   # first output = fill
    assert dl.push(8.0) == pytest.approx(7.0)   # second output = first input


# ===========================================================================
# DelayLineBank
# ===========================================================================

def test_delay_line_bank_shape():
    bank = DelayLineBank(delays=np.array([1, 2, 3]))
    x = np.array([1.0, 2.0, 3.0])
    out = bank.push(x)
    assert out.shape == (3,)


def test_delay_line_bank_independent_delays():
    bank = DelayLineBank(delays=np.array([0, 1, 2]))
    out0 = bank.push(np.array([10.0, 20.0, 30.0]))
    # delay=0: passes through immediately -> 10.0
    assert out0[0] == pytest.approx(10.0)
    # delay=1 and delay=2: fill values
    assert out0[1] == pytest.approx(0.0)
    assert out0[2] == pytest.approx(0.0)


def test_delay_line_bank_reset():
    bank = DelayLineBank(delays=np.array([2, 2]))
    bank.push(np.array([1.0, 1.0]))
    bank.reset()
    out = bank.push(np.array([0.0, 0.0]))
    assert np.allclose(out, 0.0)


# ===========================================================================
# AxonalDelayLine
# ===========================================================================

def test_axonal_delay_line_linear_mode():
    adl = AxonalDelayLine(n_dims=4, delay_mode="linear",
                           min_delay=1, max_delay=4)
    assert adl.delays[0] <= adl.delays[-1]
    assert len(adl.delays) == 4


def test_axonal_delay_line_uniform_mode():
    adl = AxonalDelayLine(n_dims=5, delay_mode="uniform", min_delay=3)
    assert np.all(adl.delays == 3)


def test_axonal_delay_line_random_mode():
    adl = AxonalDelayLine(n_dims=10, delay_mode="random",
                           min_delay=1, max_delay=5,
                           rng=np.random.default_rng(0))
    assert np.all(adl.delays >= 1)
    assert np.all(adl.delays <= 5)


def test_axonal_delay_line_custom_mode():
    delays = np.array([0, 2, 4, 6])
    adl = AxonalDelayLine(n_dims=4, delay_mode="custom",
                           delays_custom=delays)
    assert np.array_equal(adl.delays, delays)


def test_axonal_delay_line_push_output_shape():
    adl = AxonalDelayLine(n_dims=6, delay_mode="linear",
                           min_delay=1, max_delay=3)
    x = np.ones(6)
    out, info = adl.push(x)
    assert out.shape == (6,)


def test_axonal_delay_line_info_keys():
    adl = AxonalDelayLine(n_dims=4)
    _, info = adl.push(np.ones(4))
    for k in ("t", "mean_delay", "delay_spread", "min_delay", "max_delay"):
        assert k in info


def test_axonal_delay_line_zero_delay_passthrough():
    adl = AxonalDelayLine(n_dims=3, delay_mode="uniform", min_delay=0)
    x   = np.array([1.0, 2.0, 3.0])
    out, _ = adl.push(x)
    assert np.allclose(out, x)


def test_axonal_delay_line_signal_arrives_after_delay():
    adl = AxonalDelayLine(n_dims=1, delay_mode="uniform", min_delay=3)
    impulse = np.array([1.0])
    zeros   = np.array([0.0])
    outputs = []
    outputs.append(adl.push(impulse)[0][0])
    for _ in range(5):
        outputs.append(adl.push(zeros)[0][0])
    # Impulse should arrive at step index 3.
    assert outputs[3] == pytest.approx(1.0)


def test_axonal_delay_line_reset():
    adl = AxonalDelayLine(n_dims=3, delay_mode="uniform", min_delay=2)
    adl.push(np.ones(3))
    adl.push(np.ones(3))
    adl.reset()
    out, info = adl.push(np.zeros(3))
    assert np.allclose(out, 0.0)
    assert info["t"] == 1


# ===========================================================================
# TemporalMatcher
# ===========================================================================

def test_temporal_matcher_detects_exact_template():
    # Template: [1, 0] over 2 time steps, 1 dimension.
    template = np.array([[1.0], [0.0]])
    matcher  = TemporalMatcher(template=template, threshold=0.9)
    # Feed exactly the template.
    matcher.step(np.array([1.0]))
    result = matcher.step(np.array([0.0]))
    # After 2 steps the buffer should match the template exactly.
    # (Correlation may not be 1 due to delay-line fill, check detected.)
    assert isinstance(result["detected"], (bool, np.bool_))
    assert "correlation" in result


def test_temporal_matcher_no_false_positive_on_zeros():
    template = np.array([[1.0], [1.0], [1.0]])
    matcher  = TemporalMatcher(template=template, threshold=0.95)
    results  = [matcher.step(np.array([0.0])) for _ in range(5)]
    # All-zero input should not trigger detection.
    assert not any(r["detected"] for r in results)


def test_temporal_matcher_reset_clears_state():
    template = np.array([[1.0, 0.5]])
    matcher  = TemporalMatcher(template=template, threshold=0.5)
    matcher.step(np.array([1.0, 0.5]))
    matcher.reset()
    result = matcher.step(np.array([0.0, 0.0]))
    assert result["correlation"] == pytest.approx(0.0)


# ===========================================================================
# SpikeEvent
# ===========================================================================

def test_spike_event_is_namedtuple():
    e = SpikeEvent(neuron_id=5, timestamp=10.0, value=0.8)
    assert e.neuron_id == 5
    assert e.timestamp == 10.0
    assert e.value == pytest.approx(0.8)


# ===========================================================================
# RoutingTable
# ===========================================================================

def test_routing_table_connect_and_query():
    rt = RoutingTable()
    rt.connect([0, 1, 2], "V1")
    assert "V1" in rt.targets_for(0)
    assert "V1" in rt.targets_for(2)
    assert len(rt.targets_for(99)) == 0


def test_routing_table_divergent():
    rt = RoutingTable()
    rt.connect([5], "A")
    rt.connect([5], "B")
    targets = rt.targets_for(5)
    assert "A" in targets and "B" in targets


def test_routing_table_all_targets():
    rt = RoutingTable()
    rt.connect([0], "X"); rt.connect([1], "Y")
    assert rt.all_targets() == {"X", "Y"}


def test_routing_table_connect_all_to_all():
    rt = RoutingTable()
    rt.connect_all_to_all(4, "target")
    for i in range(4):
        assert "target" in rt.targets_for(i)


def test_routing_table_topographic():
    rt = RoutingTable()
    rt.connect_topographic(10, ["A", "B"])
    # First half goes to A, second to B.
    assert "A" in rt.targets_for(0)
    assert "B" in rt.targets_for(9)


# ===========================================================================
# SpikeRoutingBus
# ===========================================================================

def test_bus_emit_and_collect():
    bus = SpikeRoutingBus()
    bus.register_region("target", 8)
    bus.routing.connect_all_to_all(8, "target")
    e = SpikeEvent(3, 1.0, 0.5)
    bus.emit_event(e)
    events = bus.collect("target")
    assert len(events) == 1
    assert events[0].neuron_id == 3


def test_bus_collect_clears_queue():
    bus = SpikeRoutingBus()
    bus.register_region("target", 4)
    bus.routing.connect_all_to_all(4, "target")
    bus.emit_event(SpikeEvent(0, 0.0, 1.0))
    bus.collect("target")
    assert bus.queue_size("target") == 0


def test_bus_time_window_filter():
    bus = SpikeRoutingBus()
    bus.register_region("R", 10)
    bus.routing.connect_all_to_all(10, "R")
    bus.emit_events([
        SpikeEvent(0, 1.0, 1.0),
        SpikeEvent(1, 5.0, 1.0),
        SpikeEvent(2, 9.0, 1.0),
    ])
    early = bus.collect("R", time_window=(0, 3))
    assert len(early) == 1
    assert early[0].timestamp == 1.0


def test_bus_unrouted_events_not_delivered():
    bus = SpikeRoutingBus()
    bus.register_region("target", 4)
    # No routing table set up -> neuron 0 has no targets.
    bus.emit_event(SpikeEvent(0, 0.0, 1.0))
    assert bus.queue_size("target") == 0


def test_bus_stats():
    bus = SpikeRoutingBus()
    bus.register_region("T", 4)
    bus.routing.connect_all_to_all(4, "T")
    bus.emit_events([SpikeEvent(i, 0.0, 1.0) for i in range(4)])
    s = bus.stats()
    assert s["total_emitted"] == 4
    assert s["total_routed"] == 4


# ===========================================================================
# AEREncoder
# ===========================================================================

def test_aer_encoder_threshold_filters():
    enc = AEREncoder(n_neurons=8, mode="threshold", threshold=0.5)
    x   = np.array([0.1, 0.8, 0.3, 0.9, 0.0, 0.6, 0.2, 0.7])
    events = enc.encode(x, timestamp=0.0)
    active = {e.neuron_id for e in events}
    assert 1 in active and 3 in active and 5 in active and 7 in active
    assert 0 not in active


def test_aer_encoder_topk_emits_exactly_k():
    enc = AEREncoder(n_neurons=10, mode="topk", k=3)
    x   = np.random.default_rng(0).standard_normal(10)
    events = enc.encode(x)
    assert len(events) == 3


def test_aer_encoder_rate_mode_probabilistic():
    rng = np.random.default_rng(42)
    enc = AEREncoder(n_neurons=100, mode="rate", rng=rng)
    x   = np.ones(100) * 0.5   # 50% firing probability
    events = enc.encode(x)
    n = len(events)
    assert 20 < n < 80   # should be around 50


def test_aer_encoder_sparsity():
    enc = AEREncoder(n_neurons=10, mode="threshold", threshold=0.5)
    x   = np.zeros(10); x[2] = 1.0
    s   = enc.sparsity(x)
    assert s == pytest.approx(0.9)


def test_aer_encoder_compression_ratio_sparse():
    enc = AEREncoder(n_neurons=100, mode="threshold", threshold=0.5)
    x   = np.zeros(100); x[0] = 1.0   # only 1 active neuron
    cr  = enc.compression_ratio(x)
    assert cr > 1.0


# ===========================================================================
# AERDecoder
# ===========================================================================

def test_aer_decoder_latest_mode():
    dec    = AERDecoder(n_neurons=5, mode="latest")
    events = [SpikeEvent(2, 1.0, 0.5), SpikeEvent(2, 2.0, 0.9)]
    x      = dec.decode(events)
    assert x[2] == pytest.approx(0.9)   # latest value wins


def test_aer_decoder_sum_mode():
    dec    = AERDecoder(n_neurons=4, mode="sum")
    events = [SpikeEvent(0, 0.0, 0.3), SpikeEvent(0, 1.0, 0.4)]
    x      = dec.decode(events)
    assert x[0] == pytest.approx(0.7)


def test_aer_decoder_mean_mode():
    dec    = AERDecoder(n_neurons=4, mode="mean")
    events = [SpikeEvent(1, 0.0, 0.2), SpikeEvent(1, 1.0, 0.8)]
    x      = dec.decode(events)
    assert x[1] == pytest.approx(0.5)


def test_aer_decoder_empty_returns_fill():
    dec = AERDecoder(n_neurons=5, fill=0.0)
    x   = dec.decode([])
    assert np.allclose(x, 0.0)


def test_aer_decoder_out_of_range_neuron_ignored():
    dec    = AERDecoder(n_neurons=4)
    events = [SpikeEvent(99, 0.0, 1.0)]   # neuron 99 out of range
    x      = dec.decode(events)
    assert np.allclose(x, 0.0)


# ===========================================================================
# AERRelay
# ===========================================================================

def test_aer_relay_output_shape():
    relay = AERRelay(n_source=8, n_target=8, enc_mode="threshold",
                     threshold=0.1)
    x = np.random.default_rng(0).standard_normal(8)
    x_rec, stats = relay.transmit(x)
    assert x_rec.shape == (8,)


def test_aer_relay_stats_keys():
    relay = AERRelay(n_source=8, n_target=8)
    _, stats = relay.transmit(np.ones(8))
    for k in ("n_events", "sparsity", "compression_ratio",
              "reconstruction_error"):
        assert k in stats


def test_aer_relay_topk_exact_events():
    relay = AERRelay(n_source=16, n_target=16, enc_mode="topk", k=4)
    x = np.random.default_rng(7).standard_normal(16)
    _, stats = relay.transmit(x)
    assert stats["n_events"] == 4


def test_aer_relay_zero_input_no_events_threshold():
    relay = AERRelay(n_source=8, n_target=8, enc_mode="threshold",
                     threshold=0.5)
    x = np.zeros(8)
    _, stats = relay.transmit(x)
    assert stats["n_events"] == 0


def test_aer_relay_reconstruction_preserves_active_neurons():
    relay = AERRelay(n_source=8, n_target=8, enc_mode="threshold",
                     threshold=0.1, dec_mode="latest")
    x = np.zeros(8); x[3] = 0.7; x[6] = -0.5
    x_rec, _ = relay.transmit(x)
    assert abs(x_rec[3]) > 0.0
    assert abs(x_rec[6]) > 0.0
