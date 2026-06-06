"""Tests for the neurons/ package (LIF, STDP, Hopfield, predictive coding)."""

import numpy as np
import pytest


# ============================================================================
# LIF
# ============================================================================

from qbit_simulator.neurons.lif import (
    LIFNeuron, LIFPopulation, SNN,
    make_pure_input_snn, firing_rates, spike_raster,
    inter_spike_intervals, coefficient_of_variation,
    poisson_input_current,
)


def test_lif_subthreshold_no_fire():
    n = LIFNeuron()
    fired = sum(int(n.step(I=0.3, t=t)) for t in range(50))
    assert fired == 0


def test_lif_suprathreshold_fires_repeatedly():
    n = LIFNeuron()
    fired = sum(int(n.step(I=2.0, t=t)) for t in range(100))
    assert fired >= 3


def test_lif_refractory_period_enforced():
    """After a spike, the neuron stays at V_reset for t_refrac steps."""
    n = LIFNeuron(t_refrac=5)
    n.V = 0.99
    fired_at = []
    for t in range(30):
        if n.step(I=3.0, t=t):
            fired_at.append(t)
    # Consecutive spikes must be ≥ t_refrac + 1 apart.
    for a, b in zip(fired_at, fired_at[1:]):
        assert b - a > 5


def test_lif_refractory_gap_equals_t_refrac_plus_one():
    """Regression: t_refrac=N gives min gap = N+1 with infinite drive.

    Convention: t_refrac is the number of FULLY-BLOCKED steps after a
    spike (so the spike step is followed by t_refrac silent steps).
    Previously the code had an off-by-one giving gap = N.
    """
    n = LIFNeuron(t_refrac=3)
    n.V = 1.5     # primed to spike at t=0
    fired_at = [t for t in range(15) if n.step(I=100.0, t=t)]
    gaps = [fired_at[i + 1] - fired_at[i] for i in range(len(fired_at) - 1)]
    assert all(g == 4 for g in gaps), f"gaps {gaps} != [4, 4, ...]"


def test_lif_population_refractory_gap_equals_t_refrac_plus_one():
    """Vectorized version: same regression guard."""
    pop = LIFPopulation(n=1, t_refrac=2)
    pop.V[0] = 1.5
    fired = [t for t in range(15)
             if pop.step(np.array([100.0]), t=t)[0]]
    gaps = [fired[i + 1] - fired[i] for i in range(len(fired) - 1)]
    assert all(g == 3 for g in gaps), f"gaps {gaps} != [3, 3, ...]"


def test_lif_population_independence_no_input():
    pop = LIFPopulation(n=5)
    spikes = pop.step(np.zeros(5), t=0)
    assert not spikes.any()


def test_lif_population_reset():
    pop = LIFPopulation(n=3)
    pop.step(np.full(3, 2.0), t=0)
    pop.reset()
    assert (pop.V == pop.V_rest).all()


def test_snn_no_connections_pure_input():
    snn = make_pure_input_snn(4)
    result = snn.run(external_current=np.full(4, 2.0), n_steps=50)
    assert result["total_spikes"] > 0
    # All neurons fire at the same rate.
    rates = result["rates"]
    assert np.allclose(rates, rates[0], atol=0.05)


def test_snn_rejects_bad_weights_shape():
    W = np.zeros((3, 4))
    with pytest.raises(ValueError):
        SNN(weights=W)


def test_firing_rate_window():
    """A moving-window firing rate returns the right number of bins."""
    sh = np.zeros((20, 3), dtype=bool)
    sh[::4, :] = True
    rates = firing_rates(sh, window=5)
    assert rates.shape == (16, 3)


def test_isi_for_regular_spiking():
    sh = np.zeros((20, 1), dtype=bool)
    sh[[2, 7, 12, 17], 0] = True
    isis = inter_spike_intervals(sh, neuron=0)
    assert np.array_equal(isis, [5, 5, 5])


def test_cv_zero_for_regular_spiking():
    sh = np.zeros((20, 1), dtype=bool)
    sh[[2, 7, 12, 17], 0] = True
    assert coefficient_of_variation(sh, neuron=0) == 0.0


def test_raster_returns_string():
    sh = np.zeros((10, 3), dtype=bool)
    sh[5, 1] = True
    txt = spike_raster(sh)
    assert "|" in txt and "." in txt


def test_poisson_input_current_callable():
    rng = np.random.default_rng(0)
    fn = poisson_input_current(n=5, rate=0.5, rng=rng)
    I = fn(0)
    assert I.shape == (5,)
    # Each component is 0 or 1.
    assert set(np.unique(I)).issubset({0.0, 1.0})


# ============================================================================
# STDP
# ============================================================================

from qbit_simulator.neurons.stdp import (
    STDPRule, STDPTraces, pairwise_stdp_update, train_with_stdp,
)


def test_stdp_potentiation_for_pre_before_post():
    rule = STDPRule()
    assert rule.delta_w(5.0) > 0
    assert rule.delta_w(50.0) > 0


def test_stdp_depression_for_post_before_pre():
    rule = STDPRule()
    assert rule.delta_w(-5.0) < 0


def test_stdp_zero_at_simultaneous():
    rule = STDPRule()
    assert rule.delta_w(0.0) == 0.0


def test_stdp_decays_with_distance():
    rule = STDPRule()
    assert abs(rule.delta_w(100.0)) < abs(rule.delta_w(5.0))


def test_stdp_clip_respects_bounds():
    rule = STDPRule(w_min=0.0, w_max=1.0)
    assert rule.clip(2.0) == 1.0
    assert rule.clip(-0.5) == 0.0
    assert rule.clip(0.5) == 0.5


def test_pairwise_stdp_simple():
    rule = STDPRule()
    # Pre at 0, post at 5 → potentiation only.
    delta = pairwise_stdp_update(np.array([0]), np.array([5]), rule)
    assert delta > 0


def test_train_with_stdp_potentiation():
    """Pre @ 2, post @ 4 → weight increases."""
    rule = STDPRule()
    pre = np.zeros((10, 1), dtype=bool); pre[2, 0] = True
    post = np.zeros((10, 1), dtype=bool); post[4, 0] = True
    w_init = np.array([[0.5]])
    w_new, _ = train_with_stdp(w_init.copy(), pre, post, rule)
    assert w_new[0, 0] > w_init[0, 0]


def test_train_with_stdp_depression():
    """Post @ 2, pre @ 4 → weight decreases."""
    rule = STDPRule()
    pre = np.zeros((10, 1), dtype=bool); pre[4, 0] = True
    post = np.zeros((10, 1), dtype=bool); post[2, 0] = True
    w_init = np.array([[0.5]])
    w_new, _ = train_with_stdp(w_init.copy(), pre, post, rule)
    assert w_new[0, 0] < w_init[0, 0]


def test_stdp_traces_decay():
    rule = STDPRule(tau_plus=10.0)
    tr = STDPTraces(n_pre=3, n_post=2, rule=rule)
    tr.pre_trace = np.array([1.0, 0.5, 0.0])
    tr.step_decay()
    assert tr.pre_trace[0] < 1.0
    assert tr.pre_trace[1] < 0.5


# ============================================================================
# Hopfield
# ============================================================================

from qbit_simulator.neurons.hopfield import (
    HopfieldNetwork, pattern_overlap, corrupt_pattern, random_pattern,
    estimate_capacity,
)


def test_hopfield_stores_pattern_as_fixed_point():
    """A stored pattern shouldn't change under one update sweep."""
    rng = np.random.default_rng(0)
    n = 20
    p = random_pattern(n, rng)
    net = HopfieldNetwork(n)
    net.store_patterns(p[np.newaxis, :])
    out = net.update_async(p, rng=rng)
    assert np.array_equal(out, p)


def test_hopfield_recovers_corrupted_pattern():
    rng = np.random.default_rng(0)
    n = 40
    p = random_pattern(n, rng)
    net = HopfieldNetwork(n)
    net.store_patterns(p[np.newaxis, :])
    probe = corrupt_pattern(p, p_flip=0.1, rng=rng)
    r = net.retrieve(probe, max_iter=20, rng=rng)
    assert pattern_overlap(r["retrieved_state"], p) > 0.9


def test_hopfield_rejects_non_pm_one_patterns():
    net = HopfieldNetwork(5)
    with pytest.raises(ValueError):
        net.store_patterns(np.array([[0, 1, 0, 1, 0]]))


def test_hopfield_energy_decreases_under_updates():
    """Asynchronous updates can only decrease (or hold) energy."""
    rng = np.random.default_rng(0)
    n = 30
    patterns = np.array([random_pattern(n, rng) for _ in range(3)])
    net = HopfieldNetwork(n)
    net.store_patterns(patterns)
    probe = corrupt_pattern(patterns[0], 0.2, rng)
    E_prev = net.energy(probe)
    s = probe.copy()
    for _ in range(10):
        s = net.update_async(s, rng=rng)
        E_now = net.energy(s)
        assert E_now <= E_prev + 1e-10
        E_prev = E_now


def test_hopfield_capacity_drops_above_threshold():
    """Many random patterns → retrieval fails."""
    rng = np.random.default_rng(0)
    cap = estimate_capacity(n=40, n_patterns_list=[2, 20], p_flip=0.1,
                              n_trials=10, rng=rng)
    assert cap[2] > 0.7
    assert cap[20] < 0.5    # well above 0.14·N = 5.6


def test_pattern_overlap_self_is_one():
    rng = np.random.default_rng(0)
    p = random_pattern(15, rng)
    assert pattern_overlap(p, p) == 1.0


def test_pattern_overlap_anti_correlated_is_minus_one():
    rng = np.random.default_rng(0)
    p = random_pattern(15, rng)
    assert pattern_overlap(p, -p) == -1.0


# ============================================================================
# Predictive coding
# ============================================================================

from qbit_simulator.neurons.predictive_coding import (
    PredictiveCodingLayer, PredictiveCodingNetwork,
    free_energy, train_predictive_coding,
)


def test_pc_network_layer_count():
    net = PredictiveCodingNetwork(layer_sizes=[4, 2, 1])
    assert len(net.layers) == 2


def test_pc_infer_produces_states_and_errors():
    net = PredictiveCodingNetwork(layer_sizes=[3, 2])
    out = net.infer(np.array([1.0, 0.5, -0.2]), n_iter=20)
    assert "states" in out and "errors" in out
    assert len(out["states"]) == len(net.layers)
    assert "free_energy" in out
    assert len(out["free_energy"]) == 20


def test_pc_infer_rejects_wrong_input_shape():
    net = PredictiveCodingNetwork(layer_sizes=[3, 2])
    with pytest.raises(ValueError):
        net.infer(np.array([1.0, 0.5]))


def test_pc_training_decreases_free_energy():
    rng = np.random.default_rng(0)
    net = PredictiveCodingNetwork(layer_sizes=[4, 2])
    data = rng.normal(size=(10, 4))
    hist = train_predictive_coding(
        net, data, n_epochs=5, n_iter_per_sample=20,
        lr_x=0.1, lr_w=0.05, rng=rng,
    )
    F = hist["mean_free_energy_per_epoch"]
    assert F[-1] < F[0]


def test_pc_predict_top_down_shape():
    net = PredictiveCodingNetwork(layer_sizes=[4, 2, 1])
    out = net.predict_top_down(np.array([1.0]))
    assert out.shape == (4,)


def test_pc_predict_top_down_rejects_wrong_shape():
    net = PredictiveCodingNetwork(layer_sizes=[4, 2, 1])
    with pytest.raises(ValueError):
        net.predict_top_down(np.array([1.0, 0.5]))


def test_free_energy_starts_at_zero_for_uninit():
    net = PredictiveCodingNetwork(layer_sizes=[3, 2])
    # No inference run yet → errors are zero arrays.
    assert free_energy(net) == 0.0
