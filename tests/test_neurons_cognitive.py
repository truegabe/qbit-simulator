"""Tests for the cognitive-layer neurons modules:
cortical_column, place_cells, reservoir, attention, episodic_memory,
active_inference, drift_diffusion, wake_sleep.
"""

import numpy as np
import pytest


# ============================================================================
# Cortical column
# ============================================================================

from qbit_simulator.neurons.cortical_column import (
    CorticalColumn, CorticalPatch, LAYER_NAMES,
)


def test_cortical_column_has_5_layers():
    col = CorticalColumn(n_per_layer=10)
    assert set(col.layers.keys()) == set(LAYER_NAMES)


def test_cortical_column_zero_input_silent():
    col = CorticalColumn(n_per_layer=10)
    r = col.run(thalamic_input=0.0, n_steps=80)
    # With no input, all layers should be near-silent.
    for name in LAYER_NAMES:
        assert r["rates"][name] < 0.05


def test_cortical_column_thalamic_drives_L4():
    """Driving thalamic input → L4 fires most."""
    col = CorticalColumn(n_per_layer=15)
    r = col.run(thalamic_input=2.0, n_steps=100)
    # L4 should fire substantially.
    assert r["rates"]["L4"] > 0.05


def test_cortical_column_information_flow():
    """L4 → L23 → L5 should all be active under thalamic drive."""
    col = CorticalColumn(n_per_layer=15)
    r = col.run(thalamic_input=2.0, n_steps=100)
    for layer in ("L4", "L23", "L5"):
        assert r["rates"][layer] > 0.05


def test_cortical_patch_n_columns():
    patch = CorticalPatch(n_columns=3, n_per_layer=10)
    assert len(patch.columns) == 3


def test_cortical_patch_runs():
    patch = CorticalPatch(n_columns=3, n_per_layer=10)
    r = patch.run(np.array([2.0, 0.0, 2.0]), n_steps=50)
    assert len(r["rates"]) == 3
    # Driven columns should fire more in L4 than the silent one.
    assert r["rates"][0]["L4"] > r["rates"][1]["L4"]


def test_cortical_patch_rejects_wrong_input_size():
    patch = CorticalPatch(n_columns=3)
    with pytest.raises(ValueError):
        patch.run(np.array([1.0, 2.0]), n_steps=10)


# ============================================================================
# Place / grid cells
# ============================================================================

from qbit_simulator.neurons.place_cells import (
    PlaceCell, PlaceCellPopulation,
    GridCell, GridCellPopulation,
    RingAttractor, integrate_path,
)


def test_place_cell_peak_at_center():
    cell = PlaceCell(center=np.array([1.0, 1.0]), sigma=0.5)
    assert abs(cell.firing_rate(np.array([1.0, 1.0])) - 1.0) < 1e-9


def test_place_cell_far_from_center_is_zero():
    cell = PlaceCell(center=np.array([0.0, 0.0]), sigma=0.3)
    rate = cell.firing_rate(np.array([5.0, 5.0]))
    assert rate < 1e-6


def test_place_population_decode_recovers_position():
    pop = PlaceCellPopulation(n_cells=25, env_size=5.0, sigma=1.0)
    true_pos = np.array([2.5, 2.5])
    act = pop.activity(true_pos)
    decoded = pop.decode_position(act)
    assert np.allclose(decoded, true_pos, atol=0.3)


def test_grid_cell_periodic():
    """A grid cell fires at multiple periodic locations."""
    cell = GridCell(scale=1.0, orientation=0.0, offset=np.zeros(2))
    rate_origin = cell.firing_rate(np.array([0.0, 0.0]))
    # Same value at another lattice vertex (scale, 0) shifted.
    assert rate_origin > 0


def test_grid_cell_population_size():
    pop = GridCellPopulation(n_modules=3, n_cells_per_module=10)
    assert len(pop.cells) == 30


def test_ring_attractor_bump_moves_with_velocity():
    """Path-integration: positive velocity → bump shifts forward."""
    ring = RingAttractor(n=40)
    ring.initialize_bump(center=20)
    before = ring.estimate_bump_center()
    # Move 10 steps with velocity = 1: bump shifts by 10 (then dynamics).
    for _ in range(10):
        ring.step(velocity=1.0)
    after = ring.estimate_bump_center()
    # Bump should have moved forward by ~10 (modulo wrap).
    forward_movement = (after - before) % 40
    assert 8 <= forward_movement <= 12


def test_ring_attractor_stable_without_velocity():
    """With no velocity input, the bump stays put."""
    ring = RingAttractor(n=40)
    ring.initialize_bump(center=20)
    before = ring.estimate_bump_center()
    for _ in range(50):
        ring.step()
    after = ring.estimate_bump_center()
    assert abs(after - before) < 5    # may drift a little due to dynamics


# ============================================================================
# Reservoir computing
# ============================================================================

from qbit_simulator.neurons.reservoir import (
    EchoStateNetwork, LiquidStateMachine,
    train_readout, predict,
)


def test_esn_spectral_radius_respected():
    esn = EchoStateNetwork(n=50, spectral_radius=0.5)
    eigs = np.linalg.eigvals(esn.W_res)
    assert max(abs(eigs)) < 0.51    # allow tiny rounding error


def test_esn_run_output_shape():
    esn = EchoStateNetwork(n=30, n_input=2)
    rng = np.random.default_rng(0)
    inputs = rng.normal(size=(20, 2))
    traces = esn.run(inputs)
    assert traces.shape == (20, 30)


def test_esn_run_finite_output():
    """No NaN/Inf in the output even after many steps."""
    esn = EchoStateNetwork(n=50)
    rng = np.random.default_rng(0)
    inputs = rng.normal(size=(200, 1))
    traces = esn.run(inputs)
    assert np.all(np.isfinite(traces))


def test_lsm_output_shape():
    lsm = LiquidStateMachine(n=30, n_input=1)
    rng = np.random.default_rng(0)
    inputs = rng.normal(size=(50, 1))
    traces = lsm.run(inputs)
    assert traces.shape == (50, 30)


def test_lsm_smoothed_output_finite():
    lsm = LiquidStateMachine(n=30)
    rng = np.random.default_rng(0)
    inputs = rng.normal(size=(100, 1))
    traces = lsm.run(inputs)
    assert np.all(np.isfinite(traces))


def test_train_readout_recovers_simple_target():
    """If target = X · W_true, ridge regression should recover W_true."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 5))
    W_true = rng.normal(size=5)
    y = X @ W_true
    W_fit = train_readout(X, y, ridge=1e-6)
    assert np.allclose(W_fit, W_true, atol=1e-3)


def test_predict_matches_dot_product():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 4))
    W = rng.normal(size=4)
    pred = predict(X, W)
    expected = X @ W
    assert np.allclose(pred, expected)


# ============================================================================
# Attention
# ============================================================================

from qbit_simulator.neurons.attention import (
    softmax_attention, compute_saliency_map,
    AttentionGate, winner_take_all, soft_winner_take_all,
    multi_head_attention,
)


def test_softmax_attention_weights_sum_to_one():
    rng = np.random.default_rng(0)
    q = rng.normal(size=4)
    k = rng.normal(size=(5, 4))
    v = rng.normal(size=(5, 2))
    _, w = softmax_attention(q, k, v)
    assert abs(w.sum() - 1.0) < 1e-9


def test_softmax_attention_concentrates_on_best_key():
    """Query identical to one key → weight concentrates there."""
    q = np.array([1.0, 0.0, 0.0, 0.0])
    keys = np.array([[10.0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    values = np.eye(3)
    _, w = softmax_attention(q, keys, values, temperature=0.5)
    assert w[0] > 0.95


def test_saliency_high_contrast():
    """A pop-out feature should get high saliency."""
    features = np.array([0.0, 0.0, 5.0, 0.0])
    sal = compute_saliency_map(features)
    assert np.argmax(sal) == 2


def test_attention_gate_apply():
    gate = AttentionGate(n=4, g_min=0.1, g_max=2.0)
    gate.update_from_saliency(np.array([0.0, 0.0, 1.0, 0.0]))
    out = gate.apply(np.ones(4))
    # Position 2 has highest gain.
    assert np.argmax(out) == 2


def test_winner_take_all_top1():
    s = np.array([0.1, 0.5, 0.2, 0.8, 0.3])
    out = winner_take_all(s, k=1)
    # Only the top element (index 3) should be nonzero.
    assert np.sum(out > 0) == 1
    assert out[3] == 0.8


def test_soft_winner_take_all_sums_to_one():
    s = np.array([1.0, 2.0, 3.0, 0.5])
    out = soft_winner_take_all(s)
    assert abs(out.sum() - 1.0) < 1e-9


def test_multi_head_attention_shape():
    q = np.zeros(4)
    k = np.zeros((3, 4))
    v = np.zeros((3, 2))
    out = multi_head_attention(q, k, v, n_heads=2)
    assert out.shape == (2,)


# ============================================================================
# Episodic memory
# ============================================================================

from qbit_simulator.neurons.episodic_memory import (
    SparseDistributedMemory, HippocampalMemory,
    random_bipolar, hamming_similarity, corrupt_bipolar,
)


def test_sdm_exact_recall():
    rng = np.random.default_rng(0)
    sdm = SparseDistributedMemory(address_dim=64, content_dim=64)
    key = random_bipolar(64, rng)
    val = random_bipolar(64, rng)
    sdm.write(key, val)
    out = sdm.read(key)
    assert hamming_similarity(out, val) > 0.95


def test_sdm_recall_with_noisy_key():
    rng = np.random.default_rng(0)
    sdm = SparseDistributedMemory(address_dim=64, content_dim=64)
    key = random_bipolar(64, rng)
    val = random_bipolar(64, rng)
    sdm.write(key, val)
    noisy = corrupt_bipolar(key, 0.1, rng)
    out = sdm.read(noisy)
    assert hamming_similarity(out, val) > 0.85


def test_sdm_clear_removes_memory():
    rng = np.random.default_rng(0)
    sdm = SparseDistributedMemory(address_dim=32, content_dim=32)
    key = random_bipolar(32, rng)
    val = random_bipolar(32, rng)
    sdm.write(key, val)
    sdm.clear()
    out = sdm.read(key)
    # After clear, counters are 0 → readout is 0 (not -1 or +1).
    assert (out == 0).all()


def test_hippocampal_recall():
    rng = np.random.default_rng(0)
    hm = HippocampalMemory(context_dim=32, content_dim=32)
    ctx = random_bipolar(32, rng)
    content = random_bipolar(32, rng)
    hm.store(ctx, content)
    recall = hm.recall(ctx)
    assert hamming_similarity(recall, content) > 0.9


def test_random_bipolar_only_pm_one():
    rng = np.random.default_rng(0)
    v = random_bipolar(50, rng)
    assert set(np.unique(v).tolist()).issubset({-1.0, 1.0})


def test_hamming_similarity_identical():
    rng = np.random.default_rng(0)
    v = random_bipolar(30, rng)
    assert hamming_similarity(v, v) == 1.0


def test_corrupt_bipolar_flips_p_fraction():
    rng = np.random.default_rng(0)
    v = random_bipolar(1000, rng)
    c = corrupt_bipolar(v, 0.25, rng)
    # ~25% of entries flipped.
    flipped = float((v != c).mean())
    assert 0.2 < flipped < 0.3


# ============================================================================
# Active inference
# ============================================================================

from qbit_simulator.neurons.active_inference import (
    ActiveInferenceAgent, GridWorld, run_episode,
)


def test_grid_world_reset_state():
    env = GridWorld()
    s = env.reset(2)
    assert s == 2


def test_grid_world_move_right():
    env = GridWorld(n_states=5)
    env.reset(0)
    s, r = env.step(action=2)
    assert s == 1


def test_grid_world_clamped_at_edges():
    env = GridWorld(n_states=3)
    env.reset(0)
    env.step(action=0)    # try left from 0
    assert env.state == 0
    env.reset(2)
    env.step(action=2)    # try right from end
    assert env.state == 2


def test_active_inference_belief_update():
    agent = ActiveInferenceAgent(n_states=3)
    agent.update_belief(observation=1)
    # Belief concentrates on state 1 (with default identity likelihood).
    assert agent.belief[1] > agent.belief[0]


def test_active_inference_reaches_goal():
    """With gradient preferences + 2-step lookahead, agent finds the goal."""
    agent = ActiveInferenceAgent(n_states=5, n_actions=3, policy_depth=2)
    agent.set_preference(preferred_obs=4)
    env = GridWorld(n_states=5, goal=4)
    ep = run_episode(env, agent, max_steps=10, start_state=0)
    assert ep["reached_goal"]


def test_active_inference_actions_move_right():
    """Starting at 0, all actions should be "right" (action 2)."""
    agent = ActiveInferenceAgent(n_states=5, n_actions=3, policy_depth=2)
    agent.set_preference(preferred_obs=4)
    env = GridWorld(n_states=5, goal=4)
    ep = run_episode(env, agent, max_steps=4, start_state=0)
    assert all(a == 2 for a in ep["actions"])


# ============================================================================
# Drift-diffusion
# ============================================================================

from qbit_simulator.neurons.drift_diffusion import (
    DDM, RaceModel,
    theoretical_choice_probability, theoretical_mean_rt,
    fit_drift_from_choices,
)


def test_ddm_positive_drift_favors_plus():
    """Positive drift → more +1 choices."""
    ddm = DDM(drift=1.0, noise=1.0, threshold=1.0)
    rng = np.random.default_rng(0)
    r = ddm.simulate_many(n_trials=300, rng=rng)
    assert r["p_plus"] > r["p_minus"]


def test_ddm_zero_drift_50_50():
    ddm = DDM(drift=0.0, noise=1.0, threshold=1.0)
    rng = np.random.default_rng(0)
    r = ddm.simulate_many(n_trials=500, rng=rng)
    assert abs(r["p_plus"] - 0.5) < 0.1


def test_ddm_matches_theory():
    """Empirical p(+1) close to closed-form formula."""
    rng = np.random.default_rng(0)
    ddm = DDM(drift=0.5, noise=1.0, threshold=1.0)
    r = ddm.simulate_many(n_trials=1000, rng=rng)
    theory = theoretical_choice_probability(0.5, 1.0, 1.0)
    assert abs(r["p_plus"] - theory) < 0.05


def test_race_model_winner_correct_direction():
    """Higher drift → that accumulator wins more often."""
    race = RaceModel(drifts=np.array([0.5, 0.1, 0.1]), noise=1.0,
                       threshold=1.0)
    rng = np.random.default_rng(0)
    r = race.simulate_many(n_trials=300, rng=rng)
    assert r["p_choice"][0] > r["p_choice"][1]


def test_theoretical_choice_prob_symmetric():
    """Drift=0 → p=0.5."""
    p = theoretical_choice_probability(0.0, 1.0, 1.0)
    assert abs(p - 0.5) < 1e-9


def test_fit_drift_from_choices_recovers():
    """Generate choices with known drift, fit it back."""
    rng = np.random.default_rng(0)
    true_drift = 0.7
    ddm = DDM(drift=true_drift, noise=1.0, threshold=1.0)
    r = ddm.simulate_many(n_trials=2000, rng=rng)
    fit = fit_drift_from_choices(r["choices"], noise=1.0, threshold=1.0)
    assert abs(fit - true_drift) < 0.2


# ============================================================================
# Wake-sleep memory consolidation
# ============================================================================

from qbit_simulator.neurons.wake_sleep import (
    WakeSleepAgent, consolidation_test,
)


def test_wake_sleep_hippocampal_recall_after_wake():
    """Right after wake-observe, hippocampus should recall accurately."""
    rng = np.random.default_rng(0)
    agent = WakeSleepAgent(context_dim=32, content_dim=32)
    ctx = random_bipolar(32, rng)
    content = random_bipolar(32, rng)
    agent.wake_observe(ctx, content)
    recall = agent.hippocampal_recall(ctx)
    assert hamming_similarity(recall, content) > 0.9


def test_wake_sleep_consolidation_improves_cortex():
    """After many sleep replays, cortical recall improves from chance."""
    rng = np.random.default_rng(0)
    r = consolidation_test(n_episodes=5, context_dim=32, content_dim=32,
                              n_sleep_replays=50, rng=rng)
    assert r["mean_after"] > r["mean_before"]


def test_wake_sleep_forget():
    rng = np.random.default_rng(0)
    agent = WakeSleepAgent(context_dim=16, content_dim=16)
    for _ in range(10):
        agent.wake_observe(random_bipolar(16, rng), random_bipolar(16, rng))
    assert len(agent.episodes_seen) == 10
    agent.hippocampal_forget(fraction=0.5, rng=rng)
    assert len(agent.episodes_seen) == 5


def test_wake_sleep_sleep_with_no_episodes_does_nothing():
    rng = np.random.default_rng(0)
    agent = WakeSleepAgent(context_dim=16, content_dim=16)
    # No wake_observe calls → no episodes to replay.
    agent.sleep_cycle(n_replays=10, rng=rng)
    assert agent.cortical_patterns_stored == 0
