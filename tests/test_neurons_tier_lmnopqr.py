"""Tests for Tier L/M/N/O/P/Q/R modules."""

import numpy as np
import pytest

from qbit_simulator.neurons.sparse_coding import SparseCoder, soft_threshold
from qbit_simulator.neurons.ica import FastICA, whiten
from qbit_simulator.neurons.slow_feature import SlowFeatureAnalysis
from qbit_simulator.neurons.info_bottleneck import InformationBottleneck
from qbit_simulator.neurons.modern_hopfield import ModernHopfield
from qbit_simulator.neurons.topology import (
    watts_strogatz, barabasi_albert, modular_network,
    clustering_coefficient, average_path_length, degree_distribution,
)
from qbit_simulator.neurons.critical_avalanche import (
    BranchingNetwork, measure_avalanches, fit_power_law_exponent,
)
from qbit_simulator.neurons.spectral_analysis import (
    spectral_radius, is_stable, random_recurrent_matrix,
)
from qbit_simulator.neurons.population_codes import (
    ProbabilisticPopulationCode, sample_counts,
)
from qbit_simulator.neurons.particle_filter import ParticleFilter
from qbit_simulator.neurons.belief_propagation import PairwiseBinaryMRF
from qbit_simulator.neurons.mean_field_boltzmann import (
    MeanFieldBoltzmann, gibbs_sample,
)
from qbit_simulator.neurons.wilson_cowan import WilsonCowan
from qbit_simulator.neurons.neural_field import NeuralField1D, mexican_hat
from qbit_simulator.neurons.kuramoto import Kuramoto
from qbit_simulator.neurons.sharp_wave_ripples import RippleReplayer, detect_ripples
from qbit_simulator.neurons.internal_models import (
    ForwardModel, InverseModel, PairedInternalModels,
)
from qbit_simulator.neurons.muscle_cpg import (
    hill_force, HalfCenterCPG, MuscleArmModel,
)
from qbit_simulator.neurons.vor import VORAgent, simulate_vor
from qbit_simulator.neurons.optimal_control import lqr, simulate_lqr
from qbit_simulator.neurons.quantum_perceptron import (
    QuantumHopfield, QuantumPerceptron, basis_state,
)
from qbit_simulator.neurons.quantum_boltzmann import QuantumBoltzmann
from qbit_simulator.neurons.quantum_reservoir import (
    QuantumReservoir, train_linear_readout,
)
from qbit_simulator.neurons.quantum_free_energy import (
    von_neumann_entropy, quantum_relative_entropy, gibbs_state,
    quantum_free_energy, classical_free_energy,
)
from qbit_simulator.neurons.two_stage_consolidation import TwoStageMemory
from qbit_simulator.neurons.shy import SHYNetwork, sleep_cycle
from qbit_simulator.neurons.prioritized_replay import PrioritizedReplayBuffer


# ---- Tier L: unsupervised ----

def test_soft_threshold_basic():
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    out = soft_threshold(x, 1.0)
    assert np.allclose(out, [-1.0, 0.0, 0.0, 0.0, 1.0])


def test_sparse_coder_loss_decreases():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 8))
    sc = SparseCoder(n_features=8, n_atoms=10, lam=0.1, rng=rng)
    losses = sc.train(X, n_iter=200)
    assert np.mean(losses[-30:]) < np.mean(losses[:30])


def test_ica_recovers_independent_sources():
    rng = np.random.default_rng(0)
    n = 500
    s1 = np.sign(np.sin(np.linspace(0, 30, n)))
    s2 = rng.uniform(-1, 1, n)
    S = np.stack([s1, s2], axis=1)
    A = np.array([[1, 0.5], [0.5, 1]])
    X = S @ A.T
    ica = FastICA(n_components=2, rng=rng)
    Y = ica.fit_transform(X)
    assert Y.shape == (n, 2)


def test_whiten_decorrelates():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 3))
    X[:, 1] = X[:, 0] + 0.1 * X[:, 1]  # correlated
    Xw, _ = whiten(X)
    cov = Xw.T @ Xw / Xw.shape[0]
    off_diag = cov - np.diag(np.diag(cov))
    assert np.max(np.abs(off_diag)) < 0.1


def test_slow_feature_recovers_slow():
    rng = np.random.default_rng(0)
    T = 500
    slow = np.sin(np.linspace(0, 2, T))
    fast = rng.standard_normal(T)
    X = np.stack([slow + 0.5 * fast, slow - 0.5 * fast], axis=1)
    sfa = SlowFeatureAnalysis(n_components=1)
    Y = sfa.fit_transform(X)
    # First slow feature should correlate with `slow`.
    corr = np.corrcoef(Y[:, 0], slow)[0, 1]
    assert abs(corr) > 0.5


def test_information_bottleneck_runs():
    rng = np.random.default_rng(0)
    p_xy = rng.uniform(size=(5, 3)); p_xy /= p_xy.sum()
    ib = InformationBottleneck(n_clusters=2, beta=2.0, rng=rng)
    ib.fit(p_xy)
    assert ib.p_t_given_x.shape == (5, 2)
    assert np.allclose(ib.p_t_given_x.sum(axis=1), 1.0)


def test_modern_hopfield_retrieves_pattern():
    P = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    mh = ModernHopfield(patterns=P, beta=5.0)
    # Query close to pattern 1.
    out = mh.retrieve(np.array([0.9, 0.1, 0.0]))
    assert np.argmax(out) == 0


# ---- Tier M: topology ----

def test_watts_strogatz_clustering_high():
    A = watts_strogatz(n=30, k=4, p=0.0)   # no rewiring → ring lattice
    cc = clustering_coefficient(A)
    assert cc > 0.3


def test_barabasi_albert_has_hubs():
    A = barabasi_albert(n=50, m=2)
    deg = degree_distribution(A)
    # Standard deviation should be high (hubs).
    assert deg.std() > 1.5


def test_modular_network_block_structure():
    A = modular_network(modules=3, size_per_module=10,
                         p_intra=0.5, p_inter=0.01)
    # Density inside first block should exceed density outside.
    intra = A[:10, :10].sum()
    inter = A[:10, 10:].sum()
    assert intra > inter


def test_average_path_length_basic():
    # Chain of 5 nodes.
    A = np.zeros((5, 5), dtype=int)
    for i in range(4):
        A[i, i + 1] = A[i + 1, i] = 1
    apl = average_path_length(A)
    assert apl > 1.5


def test_branching_critical_avalanches():
    net = BranchingNetwork(n=100, sigma=1.0, p_ext=0.005)
    sizes, durs = measure_avalanches(net, n_steps=2000)
    assert len(sizes) > 0


def test_spectral_radius_basic():
    n = 100
    W = random_recurrent_matrix(n, g=0.8)
    r = spectral_radius(W)
    assert r < 1.5


def test_is_stable_zero_matrix():
    W = np.zeros((5, 5))
    assert is_stable(W)


# ---- Tier N: probabilistic ----

def test_ppc_estimate_close_to_true():
    rng = np.random.default_rng(0)
    ppc = ProbabilisticPopulationCode(n_neurons=30)
    counts = sample_counts(ppc, s=1.5, rng=rng)
    est = ppc.estimate(counts)
    assert abs(est - 1.5) < 1.0


def test_ppc_fuse_improves_estimate():
    rng = np.random.default_rng(0)
    ppc = ProbabilisticPopulationCode(n_neurons=20)
    c1 = sample_counts(ppc, s=0.5, rng=rng)
    c2 = sample_counts(ppc, s=0.5, rng=rng)
    fused = ProbabilisticPopulationCode.fuse(c1, c2)
    e_single = ppc.estimate(c1)
    e_fused = ppc.estimate(fused)
    # Both should be close to 0.5; fused at least as accurate on average.
    assert abs(e_fused - 0.5) < 2.0


def test_particle_filter_tracks_constant():
    pf = ParticleFilter(n_particles=100, state_dim=1)
    transition = lambda x: x + 0.01 * np.random.standard_normal(x.shape)
    likelihood = lambda obs, x: np.exp(-0.5 * (obs - x[0]) ** 2)
    for _ in range(20):
        pf.predict(transition)
        pf.update(1.0, likelihood)
        pf.resample_if_needed()
    est = pf.estimate()
    assert abs(est[0] - 1.0) < 0.5


def test_belief_propagation_recovers_marginals():
    mrf = PairwiseBinaryMRF(n=3)
    mrf.h[0] = 1.0
    mrf.J = np.zeros((3, 3))
    mrf.J[0, 1] = mrf.J[1, 0] = 0.5
    mrf.J[1, 2] = mrf.J[2, 1] = 0.5
    mrf.edges = [(0, 1), (1, 2)]
    marg = mrf.loopy_bp(n_iter=30)
    # Node 0 has positive field → marginal > 0.5.
    assert marg[0] > 0.5


def test_mean_field_converges():
    mfb = MeanFieldBoltzmann(n=4)
    mfb.h = np.array([1.0, 0.5, 0.0, -0.5])
    m = mfb.mean_field()
    # m[0] should be > m[3] given fields.
    assert m[0] > m[3]


def test_gibbs_sample_consistent_with_mean_field():
    rng = np.random.default_rng(0)
    mfb = MeanFieldBoltzmann(n=3)
    mfb.h = np.array([1.0, 0.0, -1.0])
    samples = gibbs_sample(mfb, n_samples=2000, burn_in=200, rng=rng)
    mean_emp = samples.mean(axis=0)
    mean_mf  = mfb.mean_field()
    assert np.max(np.abs(mean_emp - mean_mf)) < 0.2


# ---- Tier O: oscillations ----

def test_wilson_cowan_oscillates():
    wc = WilsonCowan()
    out = wc.run(n_steps=2000, P=1.0)
    # E should fluctuate, not stay constant.
    assert out["E"].std() > 0.001


def test_neural_field_bump_persists():
    nf = NeuralField1D(L=50)
    # Brief stimulus in the middle.
    I_arr = np.zeros(50); I_arr[24:26] = 2.0
    nf.step(I_arr)
    for _ in range(20):
        nf.step(I_arr * 0.0)
    # Activity localized near center.
    assert np.argmax(nf.u) > 15 and np.argmax(nf.u) < 35


def test_mexican_hat_center_positive():
    x = np.array([0.0, 1.0, 4.0])
    k = mexican_hat(x)
    assert k[0] > 0
    assert k[2] < k[0]


def test_kuramoto_synchronizes_at_high_K():
    k_low  = Kuramoto(n=50, K=0.1, sigma_omega=0.5)
    k_high = Kuramoto(n=50, K=4.0, sigma_omega=0.5)
    rl = k_low.run(n_steps=400)
    rh = k_high.run(n_steps=400)
    assert rh[-50:].mean() > rl[-50:].mean()


def test_ripple_replayer_generates_events():
    r = RippleReplayer(sequence_length=10, p_replay=0.1)
    out = r.generate(n_steps=500)
    assert out.sum() > 0


def test_detect_ripples_finds_event():
    act = np.zeros((20, 5))
    act[5:10, :] = 1.0   # 5-step ripple
    events = detect_ripples(act, threshold=2.0, min_duration=3)
    assert len(events) >= 1


# ---- Tier P: sensorimotor ----

def test_forward_model_predicts():
    fm = ForwardModel(state_dim=2, action_dim=1, eta=0.05)
    rng = np.random.default_rng(0)
    for _ in range(200):
        s = rng.standard_normal(2)
        a = rng.standard_normal(1)
        s_next = s + 0.5 * np.concatenate([a, a])
        fm.update(s, a, s_next)
    s = np.array([1.0, 1.0]); a = np.array([1.0])
    pred = fm.predict(s, a)
    target = s + 0.5 * np.concatenate([a, a])
    assert np.max(np.abs(pred - target)) < 0.5


def test_inverse_model_learns():
    im = InverseModel(state_dim=2, action_dim=1, eta=0.05)
    rng = np.random.default_rng(0)
    for _ in range(200):
        s = rng.standard_normal(2)
        a = rng.standard_normal(1)
        s_next = s + 0.5 * np.concatenate([a, a])
        im.update(s, s_next, a)
    s = np.array([0.0, 0.0]); s_next = np.array([0.5, 0.5])
    pred = im.predict(s, s_next)
    assert abs(pred[0] - 1.0) < 0.5


def test_hill_force_basic():
    f = hill_force(activation=1.0, length=1.0, velocity=0.0)
    assert f > 0


def test_cpg_oscillates():
    cpg = HalfCenterCPG()
    out = cpg.run(n_steps=2000, dt=1.0)
    # Either neuron should rise and fall.
    diff = out[:, 0] - out[:, 1]
    assert diff.max() > 0.2 and diff.min() < -0.2


def test_muscle_arm_responds():
    arm = MuscleArmModel()
    initial = arm.theta
    for _ in range(50):
        arm.step(a_flex=1.0, a_ext=0.0, dt=0.01)
    assert arm.theta != initial


def test_vor_learns_unit_gain():
    agent = VORAgent(gain=0.3, eta=0.01)
    head_vel = np.sin(np.linspace(0, 50, 1000))
    out = simulate_vor(agent, head_vel)
    assert out["gain"][-1] > out["gain"][0]


def test_lqr_stabilizes_double_integrator():
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    Q = np.eye(2); R = np.array([[0.1]])
    K = lqr(A, B, Q, R)
    x0 = np.array([1.0, 0.0])
    traj = simulate_lqr(A, B, K, x0, n_steps=300)
    assert abs(traj[-1, 0]) < abs(x0[0])


# ---- Tier Q: quantum ----

def test_basis_state_norm():
    psi = basis_state(np.array([1, 0, 1]))
    assert np.isclose(np.abs(psi).sum(), 1.0)


def test_quantum_hopfield_retrieves():
    qh = QuantumHopfield(n_qubits=3)
    p1 = np.array([1, 0, 1])
    qh.patterns.append(p1)
    out = qh.retrieve(np.array([1, 0, 1]), beta=10.0)
    # Stored pattern should be returned (bits in {-1, +1}).
    expected = np.where(p1 > 0, 1, -1)
    assert (out == expected).all()


def test_quantum_perceptron_predict_range():
    qp = QuantumPerceptron(n_qubits=2)
    z = qp.predict(np.array([1, 0]))
    assert -1.0 <= z <= 1.0


def test_quantum_boltzmann_density_trace_one():
    qbm = QuantumBoltzmann(n_qubits=2)
    qbm.b = np.array([1.0, 0.5])
    rho = qbm.density_matrix()
    assert abs(np.real(np.trace(rho)) - 1.0) < 1e-9


def test_quantum_boltzmann_marginal_sign():
    qbm = QuantumBoltzmann(n_qubits=2, beta=4.0)
    qbm.b = np.array([1.0, -1.0])
    z = qbm.marginals()
    # Positive bias → positive ⟨Z⟩.
    assert z[0] > 0 and z[1] < 0


def test_quantum_reservoir_runs():
    qr = QuantumReservoir(n_qubits=3)
    X = np.linspace(0, 1, 10)[:, None]
    feats = qr.run(X)
    assert feats.shape == (10, 3)


def test_quantum_reservoir_readout_fits():
    rng = np.random.default_rng(0)
    qr = QuantumReservoir(n_qubits=3, rng=rng)
    X = rng.uniform(size=(50, 1))
    target = np.sin(X * 5).ravel()
    feats = qr.run(X)
    w = train_linear_readout(feats, target, reg=1e-2)
    pred = feats @ w
    err = ((pred - target) ** 2).mean()
    assert err < 1.0


def test_von_neumann_entropy_pure_zero():
    psi = np.array([1.0, 0.0])
    rho = np.outer(psi, psi.conj())
    assert von_neumann_entropy(rho) < 1e-9


def test_von_neumann_entropy_maxmix():
    rho = np.eye(4) / 4
    s = von_neumann_entropy(rho)
    assert abs(s - np.log(4)) < 1e-9


def test_quantum_relative_entropy_zero():
    rho = np.eye(2) / 2
    assert quantum_relative_entropy(rho, rho) < 1e-9


def test_gibbs_state_trace_one():
    H = np.diag([0, 1, 2, 3]).astype(float)
    rho = gibbs_state(H, beta=1.0)
    assert abs(np.real(np.trace(rho)) - 1.0) < 1e-9


def test_quantum_free_energy_finite():
    H = np.diag([0.0, 1.0, 2.0, 3.0])
    rho = gibbs_state(H, beta=1.0)
    F = quantum_free_energy(rho, H, beta=1.0)
    assert np.isfinite(F)


def test_classical_free_energy_matches():
    E = np.array([0.0, 1.0, 2.0])
    p = np.exp(-E); p /= p.sum()
    F = classical_free_energy(p, E, beta=1.0)
    assert np.isfinite(F)


# ---- Tier R: consolidation ----

def test_two_stage_memory_consolidates():
    rng = np.random.default_rng(0)
    mem = TwoStageMemory(n=20, rng=rng)
    p = (rng.uniform(size=20) > 0.5).astype(float) * 2 - 1
    mem.encode(p)
    # Before consolidation: cortex doesn't know.
    cort_before = float(np.abs(mem.W_cort).sum())
    mem.consolidate(n_replays=200)
    cort_after = float(np.abs(mem.W_cort).sum())
    assert cort_after > cort_before


def test_shy_downscales_weights():
    net = SHYNetwork(n=10)
    net.W = np.ones((10, 10))
    net.sleep_pressure = 100.0
    initial = net.total_weight()
    out = sleep_cycle(net, n_steps=50)
    assert out[-1] < initial


def test_prioritized_replay_high_error_sampled_more():
    rng = np.random.default_rng(0)
    buf = PrioritizedReplayBuffer(capacity=10, alpha=1.0, rng=rng)
    # Add 5 entries with varying td error.
    for i in range(5):
        buf.add(("transition", i), td_error=1.0 if i == 2 else 0.01)
    # Sample many times.
    counts = np.zeros(5)
    for _ in range(200):
        batch, idx, w = buf.sample(1)
        for x in batch:
            counts[x[1]] += 1
    # Index 2 (high TD error) should dominate.
    assert counts[2] > counts.sum() / 5
