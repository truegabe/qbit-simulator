"""Tests for Tier iota: cognitive-quantum hybrid modules."""

import numpy as np
import pytest

from qbit_simulator.neurons.quantum_active_inference import (
    QuantumActiveInferenceAgent, speedup_analysis,
)
from qbit_simulator.neurons.quantum_sensory_coding import (
    amplitude_encode, basis_encode, angle_encode,
    swap_test_overlap, discrimination_matrix,
    QuantumV1Filter, encode_retinal_output,
)
from qbit_simulator.neurons.quantum_credit_assignment import (
    amplitude_estimation_mean, QAEReturnEstimator,
    QuantumCreditQAgent, qae_vs_classical_scaling,
)
from qbit_simulator.neurons.quantum_belief_propagation import (
    MRFTensorNetwork, quantum_bp_marginals, brute_force_marginals,
)
from qbit_simulator.neurons.quantum_predictive_hierarchy import (
    QuantumPredictiveHierarchy,
)


# ---- Tier iota.1: Quantum active inference ----

def test_quantum_active_inference_picks_same_action_as_classical():
    agent = QuantumActiveInferenceAgent(n_states=5, n_actions=3,
                                            policy_depth=3)
    agent.set_preference(preferred_obs=4, weight=4.0, gradient=True)
    agent.belief = np.array([1.0, 0, 0, 0, 0])
    classical = agent.best_action()
    out = agent.best_action_quantum()
    assert out["action"] == classical
    # AA must amplify (prob_marked > prob_initial).
    assert out["prob_marked"] > out["prob_initial"]


def test_speedup_analysis_grows_with_depth():
    s3 = speedup_analysis(3, 3)
    s7 = speedup_analysis(3, 7)
    assert s7["theoretical_speedup"] > s3["theoretical_speedup"]


def test_quantum_active_inference_step_returns_action():
    agent = QuantumActiveInferenceAgent(n_states=4, n_actions=3,
                                            policy_depth=2)
    agent.set_preference(preferred_obs=3)
    a = agent.step_quantum(observation=0)
    assert 0 <= a < 3


# ---- Tier iota.2: Quantum sensory coding ----

def test_amplitude_encode_normalized():
    img = np.array([[1.0, 2.0], [0.5, 0.0]])
    psi = amplitude_encode(img)
    assert abs(np.linalg.norm(psi) - 1.0) < 1e-9


def test_amplitude_encode_pads_to_power_of_two():
    img = np.zeros(6)        # not power of two
    img[3] = 1.0
    psi = amplitude_encode(img)
    assert len(psi) == 8


def test_basis_encode_single_state():
    bits = np.array([1, 0, 1])
    psi = basis_encode(bits)
    # |101> = index 5 in 3-qubit register.
    assert psi[5] == 1.0
    assert np.sum(np.abs(psi) ** 2) == 1


def test_angle_encode_factorizes():
    """Two pixels at v=0.5 with scale=pi/2 -> RY(pi/4) on each qubit."""
    img = np.array([0.5, 0.5])
    psi = angle_encode(img, scale=np.pi / 2)
    # After internal normalization (divide by max=0.5), v_norm=1.0,
    # so th = pi/2 * 1.0 = pi/2 -> qubit state = [cos(pi/4), sin(pi/4)].
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    expected = np.kron([c, s], [c, s])
    assert np.allclose(psi, expected, atol=1e-6)


def test_angle_encode_normalized():
    img = np.array([0.3, 0.7, 0.5])
    psi = angle_encode(img, scale=np.pi)
    assert abs(np.linalg.norm(psi) - 1.0) < 1e-9


def test_swap_test_overlap_self_is_one():
    psi = np.array([1, 0, 0, 0], dtype=complex)
    assert swap_test_overlap(psi, psi) == 1.0


def test_swap_test_overlap_orthogonal_is_zero():
    psi = np.array([1, 0], dtype=complex)
    phi = np.array([0, 1], dtype=complex)
    assert swap_test_overlap(psi, phi) == 0.0


def test_discrimination_matrix_diagonal_one():
    psi_a = np.array([1, 0], dtype=complex)
    psi_b = np.array([1, 1], dtype=complex) / np.sqrt(2)
    M = discrimination_matrix([psi_a, psi_b])
    assert abs(M[0, 0] - 1.0) < 1e-9
    assert abs(M[1, 1] - 1.0) < 1e-9
    # Symmetric.
    assert M[0, 1] == M[1, 0]


def test_v1_filter_unitary():
    f = QuantumV1Filter(n_qubits=2)
    U = f.unitary()
    assert U.shape == (4, 4)
    assert np.allclose(U @ U.conj().T, np.eye(4), atol=1e-9)


def test_retinal_encoding_orthogonal_images_low_overlap():
    def vertical(N=8):
        img = np.zeros((N, N)); img[:, N//2] = 1.0; return img
    def horizontal(N=8):
        img = np.zeros((N, N)); img[N//2, :] = 1.0; return img
    psi_v = encode_retinal_output(vertical(), mode="amplitude")
    psi_h = encode_retinal_output(horizontal(), mode="amplitude")
    overlap = swap_test_overlap(psi_v, psi_h)
    assert overlap < 0.2     # very different images stay near-orthogonal


# ---- Tier iota.3: Quantum credit assignment ----

def test_amplitude_estimation_mean_unbiased():
    values = np.array([0.5] * 100)
    out = amplitude_estimation_mean(values, n_eval_qubits=8)
    assert abs(out["estimate"] - 0.5) < 0.05


def test_amplitude_estimation_resolution_improves_with_qubits():
    values = np.array([0.31] * 50)
    coarse = amplitude_estimation_mean(values, n_eval_qubits=3)
    fine = amplitude_estimation_mean(values, n_eval_qubits=10)
    assert fine["qae_error"] <= coarse["qae_error"]


def test_qae_return_estimator_handles_empty():
    est = QAEReturnEstimator()
    out = est.estimate(np.array([]))
    assert out["value"] == 0.0
    assert out["n_samples"] == 0


def test_quantum_credit_q_agent_learns_chain():
    rng = np.random.default_rng(0)
    agent = QuantumCreditQAgent(n_states=4, n_actions=2,
                                  alpha=0.3, eps=0.4, rng=rng)
    for _ in range(200):
        s = 0
        for t in range(30):
            a = agent.act(s)
            s_next = min(s + 1, 3) if a == 1 else max(s - 1, 0)
            r = 1.0 if s_next == 3 else 0.0
            done = s_next == 3
            agent.update(s, a, r, s_next, done)
            s = s_next
            if done: break
    # Action 1 (right) should be preferred in early states.
    assert agent.Q[0, 1] > agent.Q[0, 0]


def test_qae_vs_classical_scaling_crossover():
    """At large N quantum beats classical."""
    s_small = qae_vs_classical_scaling(true_p=0.3, n_samples=16)
    s_large = qae_vs_classical_scaling(true_p=0.3, n_samples=1024)
    assert s_large["speedup_factor"] > s_small["speedup_factor"]


# ---- Tier iota.4: Quantum belief propagation ----

def test_mps_bp_matches_brute_force_on_chain():
    mrf = MRFTensorNetwork(n=5)
    mrf.h = np.array([1.0, -0.3, 0.5, 0.0, -0.8])
    for i in range(4):
        mrf.J_edges[(i, i + 1)] = 0.4
    m_tn = quantum_bp_marginals(mrf)
    m_bf = brute_force_marginals(mrf)
    assert np.max(np.abs(m_tn - m_bf)) < 1e-10


def test_mps_bp_negative_field_pulls_down():
    mrf = MRFTensorNetwork(n=3)
    mrf.h = np.array([0.0, -2.0, 0.0])
    mrf.J_edges = {(0, 1): 0.0, (1, 2): 0.0}
    m = quantum_bp_marginals(mrf)
    assert m[1] < 0.5    # negative field → marginal < 0.5


def test_mps_bp_positive_coupling_correlates():
    """Strong positive J should propagate field."""
    mrf = MRFTensorNetwork(n=3)
    mrf.h = np.array([2.0, 0.0, 0.0])
    mrf.J_edges = {(0, 1): 2.0, (1, 2): 2.0}
    m = quantum_bp_marginals(mrf)
    assert m[0] > 0.5 and m[1] > 0.5 and m[2] > 0.5


# ---- Tier iota.5: Quantum predictive hierarchy ----

def test_hierarchy_constructs_correctly():
    qph = QuantumPredictiveHierarchy(layer_sizes=[8, 4, 2])
    assert len(qph.Ws) == 2
    assert len(qph.qpc_layers) == 2


def test_hierarchy_top_down_generation():
    qph = QuantumPredictiveHierarchy(layer_sizes=[4, 4, 2])
    activities = qph.predict_top_down(0)
    assert len(activities) == 2     # sensory + intermediate


def test_hierarchy_inference_recovers_orthogonal_latent():
    """With orthogonal Ws, the hierarchy should invert top latents."""
    rng = np.random.default_rng(0)
    qph = QuantumPredictiveHierarchy(layer_sizes=[6, 4, 2], rng=rng)
    qph.Ws[0] = np.eye(6, 4)
    qph.Ws[1] = np.eye(4, 2)
    activities = qph.predict_top_down(1)
    x = activities[0]
    out = qph.infer(x)
    assert out["maps"][-1] == 1


def test_hierarchy_returns_posteriors_per_layer():
    qph = QuantumPredictiveHierarchy(layer_sizes=[4, 4, 2])
    out = qph.infer(np.array([0.5, 0.5, 0.5, 0.5]))
    assert len(out["posteriors"]) == 2
    for p in out["posteriors"]:
        assert abs(p.sum() - 1.0) < 1e-6
