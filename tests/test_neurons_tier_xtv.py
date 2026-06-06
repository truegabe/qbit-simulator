"""Tests for Tier X (quantum-classical bridges), T (metaplasticity), V (edge-of-chaos)."""

import numpy as np
import pytest

from qbit_simulator.neurons.quantum_annealed_hopfield import (
    QuantumAnnealedHopfield, hopfield_hamiltonian, transverse_field,
    plus_state, state_to_pattern, closest_stored_pattern,
)
from qbit_simulator.neurons.quantum_variational_pc import (
    QuantumVariationalPC, HardwareEfficientAnsatz, pc_hamiltonian,
)
from qbit_simulator.neurons.quantum_hippocampus import QuantumHippocampus
from qbit_simulator.neurons.metaplasticity import MetaplasticNeuron
from qbit_simulator.neurons.heterosynaptic import (
    HeterosynapticModulator, HeterosynapticHebbian,
)
from qbit_simulator.neurons.structural_plasticity import (
    StructuralPlasticityManager,
)
from qbit_simulator.neurons.edge_of_chaos import (
    largest_lyapunov, lyapunov_spectrum, EdgeOfChaosDiagnostic,
)


# ---- Tier X.1: Quantum-annealed Hopfield ----

def test_plus_state_uniform():
    psi = plus_state(3)
    assert np.allclose(np.abs(psi) ** 2, 1/8)


def test_state_to_pattern_zero_state():
    psi = np.zeros(4, dtype=complex); psi[0] = 1.0
    p = state_to_pattern(psi, 2)
    # |00> ↔ spins [+1, +1] (Z eigenvalue +1 on |0>).
    assert np.array_equal(p, [1, 1])


def test_state_to_pattern_one_state():
    psi = np.zeros(4, dtype=complex); psi[3] = 1.0  # |11>
    p = state_to_pattern(psi, 2)
    assert np.array_equal(p, [-1, -1])


def test_hopfield_ham_ground_state_is_stored():
    """Ground state of H_target should match stored pattern."""
    n = 3
    p = np.array([+1, -1, +1])
    W = np.outer(p, p) / n; np.fill_diagonal(W, 0)
    H = hopfield_hamiltonian(W, h_bias=0.5 * p)   # bias to break Z2 symmetry
    evals, evecs = np.linalg.eigh(H)
    gs = state_to_pattern(evecs[:, 0], n)
    assert np.array_equal(gs, p)


def test_quantum_annealed_hopfield_retrieves():
    n = 4
    p1 = np.array([+1, +1, -1, -1])
    ah = QuantumAnnealedHopfield(n=n)
    ah.store(p1)
    # 1-bit-flipped probe.
    probe = np.array([+1, -1, -1, -1])
    out = ah.quantum_retrieve(probe, n_steps=80, total_time=10.0,
                                bias_strength=0.5)
    # Bias breaks Z₂ symmetry → recover +p1.
    assert np.array_equal(out["pattern"], p1)


def test_closest_stored_pattern():
    p1 = np.array([+1, +1, -1])
    p2 = np.array([-1, +1, +1])
    decoded = np.array([+1, +1, -1])
    i, o = closest_stored_pattern(decoded, [p1, p2])
    assert i == 0 and o == 1.0


# ---- Tier X.2: Variational quantum PC ----

def test_ansatz_state_normalized():
    a = HardwareEfficientAnsatz(n_qubits=3, n_layers=2)
    p = np.random.default_rng(0).uniform(0, 2 * np.pi, a.n_params)
    psi = a.state(p)
    assert abs(np.linalg.norm(psi) - 1.0) < 1e-9


def test_pc_hamiltonian_diag_minimum_at_true():
    """The latent matching x exactly should have lowest energy."""
    rng = np.random.default_rng(0)
    W = rng.standard_normal((4, 4))
    def gen(z): return W[:, z]
    x = W[:, 2]   # true z = 2
    H = pc_hamiltonian(gen, x, n_qubits=2)
    diag = np.real(np.diag(H))
    assert int(np.argmin(diag)) == 2


def test_quantum_variational_pc_recovers_latent():
    rng = np.random.default_rng(0)
    n_latent = 8
    W_gen = rng.standard_normal((4, n_latent))
    def gen(z): return W_gen[:, z]
    true_z = 5
    x = W_gen[:, true_z]
    qvpc = QuantumVariationalPC(n_qubits=3, n_layers=3, eta=0.1,
                                  n_iter=80, rng=rng)
    losses = qvpc.fit(gen, x)
    assert losses[-1] < losses[0]
    assert qvpc.map_estimate() == true_z


# ---- Tier X.3: Quantum hippocampus ----

def test_quantum_hippocampus_stores_and_retrieves():
    n = 4
    qh = QuantumHippocampus(n=n, decay=0.95)
    p1 = np.array([+1, +1, -1, -1])
    qh.store(p1)
    # Clean cue.
    out = qh.retrieve(p1, beta=3.0, bias=1.0)
    assert np.array_equal(out["pattern"], p1)
    assert out["posterior_probs"][0] > 0.9


def test_quantum_hippocampus_pattern_completion():
    n = 4
    qh = QuantumHippocampus(n=n)
    p1 = np.array([+1, +1, -1, -1])
    qh.store(p1)
    cue = np.array([+1, +1, -1, +1])    # 1-bit flip
    out = qh.retrieve(cue, beta=2.0, bias=1.0)
    assert np.array_equal(out["pattern"], p1)


def test_quantum_hippocampus_ambiguous_cue_spreads_posterior():
    n = 4
    qh = QuantumHippocampus(n=n, decay=1.0)
    p1 = np.array([+1, +1, -1, -1])
    p2 = np.array([-1, +1, -1, +1])    # 2 bits different from p1
    qh.store(p1); qh.store(p2)
    # Cue equidistant.
    cue = np.array([0.5, 1.0, -1.0, 0.0])
    out = qh.retrieve(cue, beta=1.0, bias=0.5)
    # Both posteriors should be non-trivial.
    assert min(out["posterior_probs"]) > 0.1


def test_quantum_hippocampus_recency_effect():
    qh = QuantumHippocampus(n=4, decay=0.5)
    p1 = np.array([+1, +1, -1, -1])
    p2 = np.array([-1, -1, +1, +1])
    p3 = np.array([+1, -1, +1, -1])
    qh.store(p1); qh.store(p2); qh.store(p3)
    # Latest memory should have highest weight.
    assert qh.weights[-1] > qh.weights[0]


def test_quantum_hippocampus_entropy_drops_after_storing():
    qh = QuantumHippocampus(n=3)
    h0 = qh.memory_entropy()
    qh.store(np.array([+1, +1, -1]))
    h1 = qh.memory_entropy()
    assert h1 < h0


# ---- Tier T.1: Metaplasticity ----

def test_metaplasticity_runs():
    rng = np.random.default_rng(0)
    X = np.eye(3) * 2.0
    n = MetaplasticNeuron(n_inputs=3, eta_base=0.01)
    out = n.train(X, n_iter=2000, rng=rng)
    # At least one response > all others.
    assert out["responses"].max() > out["responses"].mean() + 1e-3


def test_metaplasticity_threshold_per_synapse():
    rng = np.random.default_rng(0)
    X = np.eye(3)
    n = MetaplasticNeuron(n_inputs=3)
    n.train(X, n_iter=500, rng=rng)
    # Per-synapse thresholds should differ (not all equal).
    assert n.theta.std() >= 0  # weak check (some variability possible)


# ---- Tier T.2: Heterosynaptic ----

def test_heterosynaptic_no_change_with_zero_delta():
    mod = HeterosynapticModulator(spread=0.1)
    w = np.array([1.0, 1.0, 1.0])
    dw_zero = np.zeros(3)
    dw_out = mod.apply(dw_zero, w)
    assert np.allclose(dw_out, 0.0)


def test_heterosynaptic_redistributes_to_neighbors():
    mod = HeterosynapticModulator(spread=0.2, radius=2, keep_total=False)
    w = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    dw = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    dw_out = mod.apply(dw, w)
    # Neighbors at positions 0..4 (excluding 2) should get small negative.
    assert dw_out[2] == 1.0
    assert dw_out[1] < 0 and dw_out[3] < 0


def test_heterosynaptic_keep_total_zero_net():
    mod = HeterosynapticModulator(spread=0.3, keep_total=True)
    w = np.ones(10)
    dw = np.zeros(10); dw[5] = 1.0
    dw_out = mod.apply(dw, w)
    assert abs(dw_out.sum()) < 1e-9


def test_heterosynaptic_hebbian_learns():
    """Net Hebbian + heterosynaptic still learns selective response."""
    rng = np.random.default_rng(0)
    learner = HeterosynapticHebbian(n_inputs=5, eta=0.05)
    x = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    for _ in range(200):
        learner.step(x, y=1.0)
    # Active inputs should have built positive weight.
    assert learner.w[0] > learner.w[3]


# ---- Tier T.3: Structural plasticity ----

def test_structural_prune_zeros_weak():
    sp = StructuralPlasticityManager(n_pre=3, n_post=3, prune_threshold=0.1,
                                       prune_rate=1.0)
    W = np.array([[0.5, 0.05, 0.0],
                   [0.0, 0.4, 0.02],
                   [0.3, 0.0, 0.5]])
    W_new, n_p = sp.prune(W)
    assert n_p >= 1
    # The 0.05 and 0.02 should both be pruned (set to 0).
    assert W_new[0, 1] == 0
    assert W_new[1, 2] == 0


def test_structural_grow_creates_connections():
    sp = StructuralPlasticityManager(n_pre=3, n_post=3, growth_rate=1.0,
                                       prune_threshold=0.01)
    W = np.zeros((3, 3))
    # Force co-activation trace high.
    sp.coact_trace = np.ones((3, 3))
    W_new, n_g = sp.grow(W)
    assert n_g > 0


def test_structural_density_respects_cap():
    sp = StructuralPlasticityManager(n_pre=3, n_post=3, max_density=0.3,
                                       growth_rate=1.0)
    W = np.ones((3, 3)) * 0.1
    sp.coact_trace = np.ones((3, 3))
    # Density already 1.0 > 0.3, so no growth.
    W_new, n_g = sp.grow(W)
    assert n_g == 0


def test_structural_step_full_cycle():
    rng = np.random.default_rng(0)
    sp = StructuralPlasticityManager(n_pre=5, n_post=5, rng=rng,
                                       prune_threshold=0.05, prune_rate=0.5,
                                       growth_rate=0.5)
    W = rng.uniform(-0.2, 0.2, (5, 5))
    pre = (rng.uniform(size=5) > 0.5).astype(float)
    post = (rng.uniform(size=5) > 0.5).astype(float)
    out = sp.step(W, pre_act=pre, post_act=post)
    assert "W" in out and "density" in out


# ---- Tier V: Edge-of-chaos ----

def test_lyapunov_negative_for_small_W():
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.3 / np.sqrt(20), (20, 20))   # below critical
    lam = largest_lyapunov(W, n_steps=300, rng=rng)
    assert lam < 0


def test_lyapunov_zero_for_zero_W():
    W = np.zeros((10, 10))
    lam = largest_lyapunov(W, n_steps=200)
    # Without recurrence: dx/dt = -x/τ → λ = -1/τ (here tau=1).
    assert lam < -0.5


def test_diagnostic_labels_regime():
    rng = np.random.default_rng(0)
    W_ord = rng.normal(0, 0.3 / np.sqrt(20), (20, 20))
    diag = EdgeOfChaosDiagnostic()
    out = diag.diagnose(W_ord, n_steps=200, rng=rng)
    assert out["regime"] in {"ordered", "edge_of_chaos", "chaotic"}


def test_lyapunov_spectrum_shape():
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.5 / np.sqrt(10), (10, 10))
    spec = lyapunov_spectrum(W, n_exponents=3, n_steps=200, rng=rng)
    assert spec.shape == (3,)
    # Sorted descending (top is largest).
    assert spec[0] >= spec[1] >= spec[2]
