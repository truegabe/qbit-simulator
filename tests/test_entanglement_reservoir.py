"""Tests for EntanglementReservoir and the three Cat-4 protocols."""

import numpy as np
import pytest

from qbit_simulator.entanglement_reservoir import (
    EntangledPair, EntanglementReservoir,
    SuperdenseCodingChannel, TeleportationChannel,
    EntangledBroadcastChannel,
)


# ===========================================================================
# EntangledPair
# ===========================================================================

def test_pair_starts_coherent():
    p = EntangledPair(d=10)
    assert p.coherence == pytest.approx(1.0)
    assert p.is_usable(threshold=0.5)


def test_pair_decoherence_reduces_coherence():
    p = EntangledPair(d=10)
    for _ in range(10):
        p.decohere(dt=1.0, tau=5.0)
    assert p.coherence < 1.0
    assert p.coherence > 0.0


def test_pair_fully_decohered_not_usable():
    p = EntangledPair(d=2)
    p.coherence = 0.1
    assert not p.is_usable(threshold=0.5)


def test_pair_get_circuit_returns_copy():
    p  = EntangledPair(d=4)
    qc = p.get_circuit()
    assert qc.n == 2
    assert qc.d == 4
    # Modifying circuit does not affect the stored pair.
    qc.measure(0)
    assert p._circuit is not None


def test_pair_bell_state_is_entangled():
    """The stored Bell state should have flat marginals."""
    p   = EntangledPair(d=10)
    qc  = p.get_circuit()
    m0  = qc.state.marginal(0)
    m1  = qc.state.marginal(1)
    assert np.allclose(m0, np.ones(10) / 10, atol=1e-9)
    assert np.allclose(m1, np.ones(10) / 10, atol=1e-9)


# ===========================================================================
# EntanglementReservoir
# ===========================================================================

def test_reservoir_starts_full():
    r = EntanglementReservoir(capacity=8, d=10)
    assert r.n_usable == 8
    assert r.charge_level == pytest.approx(1.0)


def test_reservoir_spend_reduces_count():
    r = EntanglementReservoir(capacity=8, d=4)
    r.spend(3)
    assert r.n_usable == 5


def test_reservoir_spend_returns_pairs():
    r     = EntanglementReservoir(capacity=4, d=4)
    pairs = r.spend(2)
    assert len(pairs) == 2
    assert all(isinstance(p, EntangledPair) for p in pairs)


def test_reservoir_spend_raises_when_insufficient():
    r = EntanglementReservoir(capacity=2, d=4)
    with pytest.raises(RuntimeError):
        r.spend(5)


def test_reservoir_try_spend_no_error():
    r     = EntanglementReservoir(capacity=2, d=4)
    pairs = r.try_spend(10)
    assert len(pairs) == 2   # only 2 available


def test_reservoir_refresh_fills_to_capacity():
    r = EntanglementReservoir(capacity=8, d=4)
    r.spend(5)
    r.refresh()
    assert r.n_usable == 8


def test_reservoir_refresh_partial():
    r = EntanglementReservoir(capacity=8, d=4)
    r.spend(8)
    r.refresh(3)
    assert r.n_usable == 3


def test_reservoir_step_decoherence():
    r = EntanglementReservoir(capacity=4, d=4, decoherence_tau=1.0)
    for _ in range(20):
        r.step(dt=1.0)
    # With tau=1, pairs degrade quickly -- reservoir should be mostly empty.
    assert r.charge_level < 0.5


def test_reservoir_stats_keys():
    r    = EntanglementReservoir(capacity=4, d=10)
    stats = r.stats()
    for k in ("capacity", "n_usable", "charge_level", "mean_coherence",
              "d", "bits_available", "total_spent", "total_refreshed"):
        assert k in stats


def test_reservoir_bits_available():
    r = EntanglementReservoir(capacity=4, d=10)
    # 4 pairs * 2 * log2(10) bits each
    expected = 4 * 2 * np.log2(10)
    assert abs(r.stats()["bits_available"] - expected) < 1e-9


def test_reservoir_repr_contains_info():
    r = EntanglementReservoir(capacity=4, d=10)
    s = repr(r)
    assert "d=10" in s
    assert "usable" in s


# ===========================================================================
# SuperdenseCodingChannel
# ===========================================================================

def test_superdense_roundtrip_small_d():
    rng = np.random.default_rng(42)
    r   = EntanglementReservoir(capacity=20, d=4)
    ch  = SuperdenseCodingChannel(r)
    for m in range(16):   # d^2 = 16
        r.refresh()
        recovered = ch.send(m, rng)
        assert recovered == m, f"message {m} recovered as {recovered}"


def test_superdense_roundtrip_d10():
    rng = np.random.default_rng(7)
    r   = EntanglementReservoir(capacity=10, d=10)
    ch  = SuperdenseCodingChannel(r)
    for m in [0, 1, 42, 55, 99]:
        r.refresh()
        assert ch.send(m, rng) == m


def test_superdense_bits_per_use():
    r  = EntanglementReservoir(capacity=4, d=10)
    ch = SuperdenseCodingChannel(r)
    assert ch.bits_per_use == pytest.approx(2 * np.log2(10))


def test_superdense_spends_one_pair():
    r  = EntanglementReservoir(capacity=4, d=4)
    ch = SuperdenseCodingChannel(r)
    ch.send(3, np.random.default_rng(0))
    assert r.n_usable == 3


def test_superdense_message_out_of_range_raises():
    r  = EntanglementReservoir(capacity=4, d=4)
    ch = SuperdenseCodingChannel(r)
    with pytest.raises(ValueError):
        ch.encode(16)   # max is d^2 - 1 = 15 for d=4


# ===========================================================================
# TeleportationChannel
# ===========================================================================

def test_teleport_recovers_basis_state():
    rng = np.random.default_rng(0)
    r   = EntanglementReservoir(capacity=10, d=4)
    ch  = TeleportationChannel(r)
    # Teleport |0> = [1, 0, 0, 0]
    psi   = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    r.refresh()
    psi_recovered = ch.teleport(psi, rng)
    # The recovered state should have highest amplitude on index 0.
    assert np.argmax(np.abs(psi_recovered) ** 2) == 0


def test_teleport_recovers_superposition():
    rng = np.random.default_rng(5)
    r   = EntanglementReservoir(capacity=10, d=2)
    ch  = TeleportationChannel(r)
    # Teleport |+> = [1, 1] / sqrt(2)
    psi = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
    r.refresh()
    psi_rec = ch.teleport(psi, rng)
    assert psi_rec.shape == (2,)
    # Both amplitudes should be roughly equal in magnitude.
    probs = np.abs(psi_rec) ** 2
    assert abs(probs[0] - probs[1]) < 0.3


def test_teleport_spends_one_pair():
    r  = EntanglementReservoir(capacity=4, d=4)
    ch = TeleportationChannel(r)
    psi = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    ch.teleport(psi, np.random.default_rng(0))
    assert r.n_usable == 3


def test_teleport_wrong_size_raises():
    r   = EntanglementReservoir(capacity=4, d=4)
    ch  = TeleportationChannel(r)
    psi = np.array([1.0, 0.0], dtype=complex)   # wrong size (d=4 expected)
    with pytest.raises(ValueError):
        ch.send(psi)


# ===========================================================================
# EntangledBroadcastChannel
# ===========================================================================

def test_broadcast_all_receivers_agree():
    rng = np.random.default_rng(42)
    r   = EntanglementReservoir(capacity=8, d=4)
    ch  = EntangledBroadcastChannel(r)
    result = ch.broadcast(n_receivers=4, rng=rng)
    assert result["all_agree"]


def test_broadcast_outcome_in_range():
    rng = np.random.default_rng(1)
    r   = EntanglementReservoir(capacity=4, d=10)
    ch  = EntangledBroadcastChannel(r)
    result = ch.broadcast(n_receivers=3, rng=rng)
    assert 0 <= result["alice_outcome"] < 10
    for o in result["receiver_outcomes"]:
        assert 0 <= o < 10


def test_broadcast_spends_n_pairs():
    r  = EntanglementReservoir(capacity=8, d=4)
    ch = EntangledBroadcastChannel(r)
    ch.broadcast(n_receivers=3, rng=np.random.default_rng(0))
    assert r.n_usable == 5   # 8 - 3


def test_broadcast_no_pairs_raises():
    r  = EntanglementReservoir(capacity=4, d=4)
    r.spend(4)   # drain reservoir
    ch = EntangledBroadcastChannel(r)
    with pytest.raises(RuntimeError):
        ch.broadcast(n_receivers=2)


def test_broadcast_result_keys():
    r      = EntanglementReservoir(capacity=4, d=4)
    ch     = EntangledBroadcastChannel(r)
    result = ch.broadcast(n_receivers=2, rng=np.random.default_rng(0))
    for k in ("alice_outcome", "receiver_outcomes",
              "n_pairs_spent", "mean_coherence", "all_agree"):
        assert k in result


def test_broadcast_bits_per_use():
    r  = EntanglementReservoir(capacity=4, d=10)
    ch = EntangledBroadcastChannel(r)
    assert ch.bits_per_broadcast == pytest.approx(np.log2(10))


# ===========================================================================
# Integration: full sleep-wake cycle
# ===========================================================================

def test_sleep_wake_cycle():
    """Simulate: sleep (refresh) -> waking (spend via protocols) -> deplete."""
    rng = np.random.default_rng(99)
    r   = EntanglementReservoir(capacity=12, d=10,
                                  decoherence_tau=50.0)
    sdc = SuperdenseCodingChannel(r)
    bc  = EntangledBroadcastChannel(r)

    # Sleep: reservoir fully charged.
    assert r.charge_level == pytest.approx(1.0)

    # Waking: send 4 messages via superdense coding.
    messages = [0, 42, 99, 55]
    for m in messages:
        recovered = sdc.send(m, rng)
        assert recovered == m

    # Broadcast to 3 regions.
    result = bc.broadcast(n_receivers=3, rng=rng)
    assert result["all_agree"]

    # Reservoir is now partially depleted (4 + 3 = 7 pairs spent).
    assert r.n_usable == 12 - 7

    # Decoherence over time.
    for _ in range(10):
        r.step(dt=1.0)
    assert r.mean_coherence < 1.0

    # Sleep again: refresh.
    r.refresh()
    assert r.n_usable == 12
