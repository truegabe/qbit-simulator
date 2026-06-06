"""Tests for R-STDP, spike-logic gates, working memory, free-energy bridge."""

import numpy as np
import pytest


# ============================================================================
# R-STDP
# ============================================================================

from qbit_simulator.neurons.rstdp import (
    RSTDPLearner, train_with_rstdp, pulse_reward, exponential_reward,
)
from qbit_simulator.neurons.stdp import STDPRule


def test_rstdp_no_reward_no_weight_change():
    """Without reward, eligibility decays but weights don't change."""
    pre = np.zeros((20, 1), dtype=bool); pre[2, 0] = True
    post = np.zeros((20, 1), dtype=bool); post[4, 0] = True
    reward = np.zeros(20)
    w0 = np.array([[0.5]])
    w1, _ = train_with_rstdp(w0.copy(), pre, post, reward, lr=1.0)
    assert np.allclose(w1, w0, atol=1e-12)


def test_rstdp_pre_then_post_then_reward_potentiates():
    """Pre@2, post@4 → positive eligibility. Reward later → weight up."""
    pre = np.zeros((50, 1), dtype=bool); pre[2, 0] = True
    post = np.zeros((50, 1), dtype=bool); post[4, 0] = True
    reward = pulse_reward(50, [20], amplitude=1.0)
    w0 = np.array([[0.5]])
    w1, _ = train_with_rstdp(w0.copy(), pre, post, reward, lr=1.0)
    assert w1[0, 0] > 0.5


def test_rstdp_post_then_pre_then_reward_depresses():
    """Post@2, pre@4 → negative eligibility. Reward → weight down."""
    pre = np.zeros((50, 1), dtype=bool); pre[4, 0] = True
    post = np.zeros((50, 1), dtype=bool); post[2, 0] = True
    reward = pulse_reward(50, [20], amplitude=1.0)
    w0 = np.array([[0.5]])
    w1, _ = train_with_rstdp(w0.copy(), pre, post, reward, lr=1.0)
    assert w1[0, 0] < 0.5


def test_rstdp_eligibility_decays():
    """Reward that arrives LONG after pre/post should have smaller effect."""
    pre = np.zeros((100, 1), dtype=bool); pre[2, 0] = True
    post = np.zeros((100, 1), dtype=bool); post[4, 0] = True
    early_reward = pulse_reward(100, [5], 1.0)
    late_reward = pulse_reward(100, [90], 1.0)
    w_early, _ = train_with_rstdp(np.array([[0.5]]), pre, post,
                                     early_reward, tau_eligibility=10.0, lr=1.0)
    w_late, _ = train_with_rstdp(np.array([[0.5]]), pre, post,
                                    late_reward, tau_eligibility=10.0, lr=1.0)
    # Earlier reward → larger weight change.
    assert (w_early[0, 0] - 0.5) > (w_late[0, 0] - 0.5)


def test_rstdp_negative_reward_reverses_sign():
    """Pre@2, post@4 (positive eligibility) + NEGATIVE reward → weight down."""
    pre = np.zeros((50, 1), dtype=bool); pre[2, 0] = True
    post = np.zeros((50, 1), dtype=bool); post[4, 0] = True
    reward = pulse_reward(50, [20], amplitude=-1.0)
    w0 = np.array([[0.5]])
    w1, _ = train_with_rstdp(w0.copy(), pre, post, reward, lr=1.0)
    assert w1[0, 0] < 0.5


def test_pulse_reward_shape():
    r = pulse_reward(n_steps=10, pulse_times=[3, 7], amplitude=2.0)
    assert r[3] == 2.0
    assert r[7] == 2.0
    assert r[0] == 0.0


def test_exponential_reward_decays():
    r = exponential_reward(n_steps=20, pulse_times=[5],
                              amplitude=1.0, tau=5.0)
    assert r[5] == 1.0
    assert r[10] < r[5]
    assert r[15] < r[10]


def test_rstdp_learner_reset():
    rule = STDPRule()
    learner = RSTDPLearner(n_pre=2, n_post=3, rule=rule)
    learner.pre_trace[:] = 1
    learner.eligibility[:] = 0.5
    learner.reset()
    assert (learner.pre_trace == 0).all()
    assert (learner.eligibility == 0).all()


# ============================================================================
# Spike-logic gates
# ============================================================================

from qbit_simulator.neurons.spike_logic import (
    compute_AND, compute_OR, compute_NOT, compute_XOR,
    XORNetwork, truth_table,
)


def test_AND_truth_table():
    """AND should fire only on (1,1)."""
    assert compute_AND(0, 0) == 0
    assert compute_AND(0, 1) == 0
    assert compute_AND(1, 0) == 0
    assert compute_AND(1, 1) > 0.0


def test_OR_truth_table():
    """OR should fire for any input combination except (0,0)."""
    assert compute_OR(0, 0) == 0
    assert compute_OR(0, 1) > 0.0
    assert compute_OR(1, 0) > 0.0
    assert compute_OR(1, 1) > 0.0


def test_NOT_truth_table():
    """NOT fires only when input is 0."""
    assert compute_NOT(0) > 0
    assert compute_NOT(1) == 0


def test_XOR_truth_table():
    """XOR fires only on (0,1) or (1,0). Uses longer window."""
    rate_00 = compute_XOR(0, 0, n_steps=400)
    rate_01 = compute_XOR(0, 1, n_steps=400)
    rate_10 = compute_XOR(1, 0, n_steps=400)
    rate_11 = compute_XOR(1, 1, n_steps=400)
    # XOR=true (rate > 0.01); XOR=false (rate < 0.01).
    assert rate_00 < 0.01
    assert rate_11 < 0.01
    assert rate_01 > 0.01
    assert rate_10 > 0.01


def test_truth_table_helper():
    tbl = truth_table(compute_AND, arity=2)
    assert len(tbl) == 4    # 2^2 cases


def test_xor_network_intermediate_rates():
    """OR neuron should fire more than AND for any non-(0,0)."""
    net = XORNetwork(n_steps=200)
    r = net.run(0, 1)
    assert r["OR_rate"] > r["AND_rate"]
    r = net.run(1, 0)
    assert r["OR_rate"] > r["AND_rate"]


# ============================================================================
# Working memory
# ============================================================================

from qbit_simulator.neurons.working_memory import (
    BistableAttractor, WorkingMemoryBuffer, capacity_test,
)


def test_bistable_starts_silent():
    att = BistableAttractor(n=15)
    att.reset()
    att.run_for(50)
    assert att.read() < 0.01


def test_bistable_load_creates_persistent_activity():
    att = BistableAttractor(n=15)
    att.reset()
    att.load()
    assert att.read() > 0.05


def test_bistable_clear_returns_to_silent():
    att = BistableAttractor(n=15)
    att.reset()
    t = att.load()
    t = att.clear(start_t=t)
    assert att.read() < 0.01


def test_bistable_load_then_silent_run_persists():
    """After load, free-run with no input should KEEP firing."""
    att = BistableAttractor(n=15)
    att.reset()
    t = att.load()
    for k in range(100):
        att.step(0.0, t=t + k)
    assert att.read() > 0.05


def test_bistable_load_clear_load_cycles():
    """Repeated load/clear cycles should work."""
    att = BistableAttractor(n=15)
    att.reset()
    t = 0
    for cycle in range(3):
        t = att.load(start_t=t)
        assert att.read() > 0.05
        t = att.clear(start_t=t)
        assert att.read() < 0.01


def test_working_memory_buffer_independent_slots():
    """Setting one slot shouldn't affect the others."""
    buf = WorkingMemoryBuffer(n_items=3, neurons_per_item=15)
    buf.set(1)
    buf.free_run(50)
    states = buf.all_states()
    assert states[0] == 0
    assert states[1] == 1
    assert states[2] == 0


def test_working_memory_pattern_storage():
    r = capacity_test(n_items=4, pattern=[1, 0, 1, 0], n_free_steps=80,
                        neurons_per_item=15)
    assert r["match"]


def test_working_memory_all_high():
    r = capacity_test(n_items=4, pattern=[1, 1, 1, 1], n_free_steps=80,
                        neurons_per_item=15)
    assert r["match"]


def test_working_memory_all_low():
    r = capacity_test(n_items=4, pattern=[0, 0, 0, 0], n_free_steps=80,
                        neurons_per_item=15)
    assert r["match"]


def test_buffer_rejects_out_of_range_idx():
    buf = WorkingMemoryBuffer(n_items=2)
    with pytest.raises(IndexError):
        buf.set(5)
    with pytest.raises(IndexError):
        buf.get(-1)


def test_buffer_clear_specific_slot():
    buf = WorkingMemoryBuffer(n_items=3, neurons_per_item=15)
    buf.set(0); buf.set(1); buf.set(2)
    buf.free_run(50)
    buf.clear(1)
    buf.free_run(50)
    s = buf.all_states()
    assert s[0] == 1
    assert s[1] == 0
    assert s[2] == 1


# ============================================================================
# Free-energy / quantum bridge
# ============================================================================

from qbit_simulator.neurons.free_energy_bridge import (
    pc_to_potential, quantum_free_energy,
    best_classical_explanation, match_pc_with_quantum,
)
from qbit_simulator.neurons.predictive_coding import PredictiveCodingNetwork
from qbit_simulator.algorithms.ssvqe import hardware_efficient_ansatz_apply


def test_pc_to_potential_diagonal():
    """The potential should be a diagonal Hermitian matrix."""
    net = PredictiveCodingNetwork(layer_sizes=[3, 2, 2])
    net.layers[-1].W = np.eye(2)
    net.layers[0].W = np.ones((3, 2))
    x = np.array([1.0, 0.5, 0.0])
    V = pc_to_potential(net, x)
    # Off-diagonal entries must all be zero.
    off = V - np.diag(np.diag(V))
    assert np.allclose(off, 0)
    # Hermitian.
    assert np.allclose(V, V.conj().T)


def test_pc_to_potential_dimensions():
    """For top_size=4 → 2 qubits → 4×4 potential."""
    net = PredictiveCodingNetwork(layer_sizes=[3, 2, 4])
    net.layers[-1].W = np.random.default_rng(0).normal(size=(2, 4))
    net.layers[0].W = np.random.default_rng(1).normal(size=(3, 2))
    x = np.zeros(3)
    V = pc_to_potential(net, x)
    assert V.shape == (4, 4)


def test_quantum_free_energy_non_negative_for_diagonal_potential():
    """For diagonal V with non-negative entries, F ≥ 0."""
    V = np.diag([0.0, 1.0, 2.0, 3.0]).astype(complex)
    psi = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)
    F = quantum_free_energy(psi, V)
    assert F >= 0


def test_best_classical_explanation_matches_ground_truth():
    """For a simple PC network where top-state 0 generates x exactly,
    best_classical_explanation should return index 0."""
    net = PredictiveCodingNetwork(layer_sizes=[2, 2, 2])
    net.layers[-1].W = np.eye(2)
    net.layers[0].W = np.eye(2)
    x = np.array([1.0, 0.0])
    r = best_classical_explanation(net, x)
    assert r["best_index"] == 0
    assert r["best_energy"] < 1e-9


def test_match_pc_with_quantum_finds_best_explanation():
    """Quantum-variational optimizer should find the same answer as
    classical brute force."""
    net = PredictiveCodingNetwork(layer_sizes=[4, 2, 2])
    net.layers[-1].W = np.eye(2)
    net.layers[0].W = np.array([[1.0, 0.0], [0.5, 0.0], [0.0, 1.0], [0.0, 0.5]])
    x = np.array([1.0, 0.5, 0.0, 0.0])
    classical = best_classical_explanation(net, x)

    ansatz, n_p = hardware_efficient_ansatz_apply(n_qubits=1, depth=2)
    rng = np.random.default_rng(0)
    init = rng.uniform(-1, 1, size=n_p)
    result = match_pc_with_quantum(net, x, ansatz, n_qubits=1,
                                      init_params=init, use_qng=False,
                                      n_iter=30, lr=0.3)
    # Final amplitude concentrated on the classical best index.
    probs = np.abs(result["final_state"]) ** 2
    quantum_best = int(np.argmax(probs))
    assert quantum_best == classical["best_index"]


def test_match_pc_qng_path():
    """QNG path runs without crashing."""
    net = PredictiveCodingNetwork(layer_sizes=[2, 2, 2])
    net.layers[-1].W = np.eye(2)
    x = np.array([1.0, 0.0])
    ansatz, n_p = hardware_efficient_ansatz_apply(n_qubits=1, depth=1)
    init = np.zeros(n_p)
    result = match_pc_with_quantum(net, x, ansatz, n_qubits=1,
                                      init_params=init, use_qng=True,
                                      n_iter=10, lr=0.1)
    assert "free_energy" in result
    assert len(result["free_energy"]) == 11  # init + 10 iters
