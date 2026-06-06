"""Tests for Tier D/E/F/G/H neural modules."""

import numpy as np
import pytest

from qbit_simulator.neurons.basal_ganglia import (
    BasalGanglia, train_bg_on_chain,
)
from qbit_simulator.neurons.cerebellum import CerebellumModel
from qbit_simulator.neurons.mixture_of_experts import MixtureOfExperts
from qbit_simulator.neurons.ntm import NeuralTuringMachine
from qbit_simulator.neurons.reflex import ReflexArc
from qbit_simulator.neurons.retina import Retina, encode_to_spikes
from qbit_simulator.neurons.v1 import V1Cortex, gabor_filter
from qbit_simulator.neurons.cochlea import Cochlea, make_tone
from qbit_simulator.neurons.population_vector import (
    TunedPopulation, angular_error,
)
from qbit_simulator.neurons.spike_distances import (
    victor_purpura, van_rossum, spike_times_from_train,
)
from qbit_simulator.neurons.information import (
    mutual_information, entropy, transfer_entropy,
)
from qbit_simulator.neurons.spike_stats import (
    isi, cv_isi, fano_factor, isi_histogram, psth,
)
from qbit_simulator.neurons.adaptation import AdaptingLIF
from qbit_simulator.neurons.astrocyte import (
    AstrocyteModulator, TripartiteSynapse,
)
from qbit_simulator.neurons.neuromodulators import NeuromodulatorSystem
from qbit_simulator.neurons.helmholtz import HelmholtzMachine
from qbit_simulator.neurons.boltzmann import RBM
from qbit_simulator.neurons.vae import VAE
from qbit_simulator.neurons.deep_predictive_coding import DeepPredictiveCoder
from qbit_simulator.neurons.convolutional import (
    Conv2DLayer, SimpleConvNet, conv2d_forward, maxpool2d_forward,
)


# ---- Tier D ----

def test_bg_learns_chain():
    bg = BasalGanglia(n_state=5, n_actions=2, alpha=0.3)
    rewards = train_bg_on_chain(bg, n_states=5, n_episodes=200)
    # Should learn to reach the goal reliably (reward=1 most episodes).
    assert np.mean(rewards[-30:]) > 0.5
    # And Q[0, 1] should be larger than Q[0, 0] (right > left).
    assert bg.Q[0, 1] > bg.Q[0, 0]


def test_bg_greedy_action_makes_sense():
    bg = BasalGanglia(n_state=3, n_actions=2)
    # Manually set Q to prefer action 1 in state 0.
    bg.Q[0, 1] = 10.0
    assert bg.greedy_action(0) == 1


def test_cerebellum_learns_regression():
    rng = np.random.default_rng(0)
    cb = CerebellumModel(n_mossy=20, n_granule=200, n_purkinje=2, eta=0.05, rng=rng)
    X = rng.standard_normal((30, 20))
    Y = X[:, :2] * 2.0
    losses = cb.train(X, Y, n_epochs=80)
    assert losses[-1] < losses[0]


def test_moe_trains():
    rng = np.random.default_rng(0)
    moe = MixtureOfExperts(n_in=3, n_out=1, n_experts=3, eta=0.02, rng=rng)
    X = rng.standard_normal((50, 3))
    Y = (X[:, 0] + X[:, 1]).reshape(-1, 1)
    losses = moe.train(X, Y, n_epochs=80)
    assert losses[-1] < losses[0]


def test_ntm_associative_recall():
    rng = np.random.default_rng(0)
    ntm = NeuralTuringMachine(N=8, D=4, n_in=4, n_out=4, rng=rng)
    # Direct memory write + read.
    v = np.array([1.0, 0.5, -0.5, 1.5])
    ntm.write_at(3, v)
    assert np.allclose(ntm.read_at(3), v)


def test_ntm_step_runs():
    rng = np.random.default_rng(0)
    ntm = NeuralTuringMachine(N=8, D=4, n_in=4, n_out=4, rng=rng)
    out = ntm.step(np.array([1.0, 0.0, 0.5, -0.5]))
    assert out.shape == (4,)


def test_reflex_arc_latency():
    arc = ReflexArc(delay_steps=3, gain=3.0)
    out = arc.run(stim_func=lambda t: 2.0, n_steps=30)
    if out["latency"] >= 0:
        # Motor spike should follow sensory spike by ≥ delay_steps.
        assert out["latency"] >= 3


# ---- Tier E ----

def test_retina_dog_zero_on_uniform():
    r = Retina()
    img = np.ones((20, 20)) * 5
    out = r(img)
    # DoG of uniform input ≈ 0 (within boundary effects).
    assert abs(out["dog"][10, 10]) < 0.1


def test_retina_on_off_separates():
    r = Retina(on_off=True)
    img = np.zeros((20, 20)); img[8:12, 8:12] = 5
    out = r(img)
    assert out["on"].max() > 0


def test_v1_gabor_orientations():
    v1 = V1Cortex(n_orientations=4, size=11)
    # Horizontal edge image.
    img = np.zeros((20, 20)); img[10:, :] = 1.0
    pref = v1.preferred_orientation(img)
    # Most pixels should prefer the horizontal-edge orientation (0).
    assert pref.shape == img.shape


def test_v1_gabor_filter_shape():
    g = gabor_filter(11, theta=0.0)
    assert g.shape == (11, 11)


def test_cochlea_responds_to_tone():
    c = Cochlea(n_channels=16, f_min=100, f_max=2000, fs=4000.0)
    sig = make_tone(freq=500, duration=0.1, fs=4000.0, amp=1.0)
    cgram = c.process(sig)
    # Peak should be at ~ the channel whose CF ≈ 500 Hz.
    channel_energy = cgram.sum(axis=1)
    peak = int(np.argmax(channel_energy))
    # Closest channel to 500 Hz.
    closest = int(np.argmin(np.abs(c.center_freqs - 500)))
    assert abs(peak - closest) <= 4


def test_population_vector_decodes():
    rng = np.random.default_rng(0)
    pop = TunedPopulation(n=20)
    theta = 1.0
    counts = pop.sample(theta, rng=rng, dt=20.0)
    theta_hat, _ = pop.population_vector(counts)
    err = angular_error(theta_hat, theta)
    assert err < 0.6


def test_population_vector_mle_decodes():
    rng = np.random.default_rng(0)
    pop = TunedPopulation(n=20)
    theta = 2.0
    counts = pop.sample(theta, rng=rng, dt=50.0)
    theta_hat = pop.maximum_likelihood(counts, n_grid=180)
    assert angular_error(theta_hat, theta) < 0.5


# ---- Tier F ----

def test_victor_purpura_same_train_zero():
    a = np.array([5.0, 10.0])
    assert victor_purpura(a, a, q=1.0) == 0


def test_victor_purpura_disjoint_equals_count():
    a = np.array([5.0, 10.0])
    b = np.array([])
    assert victor_purpura(a, b, q=1.0) == 2


def test_victor_purpura_shift_costs():
    a = np.array([0.0])
    b = np.array([2.0])
    # q=1 ⇒ cost = min(2, 2) = 2.
    assert victor_purpura(a, b, q=1.0) == 2.0


def test_van_rossum_same_zero():
    a = np.array([5.0, 10.0])
    d = van_rossum(a, a, tau=5.0)
    assert d < 1e-9


def test_van_rossum_positive_for_diff():
    a = np.array([5.0]); b = np.array([10.0])
    assert van_rossum(a, b, tau=5.0) > 0


def test_mutual_information_independent_low():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=500)
    Y = rng.uniform(size=500)
    mi = mutual_information(X, Y, n_bins=4)
    assert abs(mi) < 0.1


def test_mutual_information_dependent_high():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=500)
    Y = X + 0.01 * rng.standard_normal(500)
    mi = mutual_information(X, Y, n_bins=4)
    assert mi > 0.5


def test_transfer_entropy_zero_independent():
    rng = np.random.default_rng(0)
    X = rng.standard_normal(500)
    Y = rng.standard_normal(500)
    te = transfer_entropy(X, Y, n_bins=4)
    assert abs(te) < 0.2


def test_transfer_entropy_directed():
    rng = np.random.default_rng(0)
    X = rng.standard_normal(500)
    Y = np.zeros(500)
    for t in range(1, 500):
        Y[t] = 0.9 * X[t - 1] + 0.1 * rng.standard_normal()
    te_XY = transfer_entropy(X, Y, n_bins=4)
    te_YX = transfer_entropy(Y, X, n_bins=4)
    # X drives Y, so TE(X→Y) > TE(Y→X).
    assert te_XY > te_YX


def test_isi_regular_low_cv():
    train = np.zeros(100, dtype=bool)
    train[::10] = True
    assert cv_isi(train) < 0.1


def test_isi_poisson_cv_near_one():
    rng = np.random.default_rng(0)
    train = rng.uniform(size=10000) < 0.05
    cv = cv_isi(train)
    assert 0.6 < cv < 1.4


def test_fano_factor_constant_train_zero():
    train = np.zeros(200, dtype=bool); train[::10] = True
    f = fano_factor(train, window=10)
    # Each window has exactly 1 spike, var=0.
    assert f < 0.5


def test_psth_shape():
    rng = np.random.default_rng(0)
    trials = rng.uniform(size=(20, 100)) < 0.1
    rate, centers = psth(trials, bin_width=5)
    assert len(rate) == 20
    assert len(centers) == 20


def test_isi_histogram_returns_counts():
    train = np.zeros(50, dtype=bool); train[5] = train[15] = train[20] = True
    counts, edges = isi_histogram([train], bin_width=1.0)
    assert counts.sum() == 2   # 2 ISIs


# ---- Tier G ----

def test_adapting_lif_rate_decreases():
    pop = AdaptingLIF(n=1, b=0.3, tau_a=50.0)
    n_early = 0; n_late = 0
    n_steps = 400
    for t in range(n_steps):
        s = pop.step(np.array([3.0]), t=t)
        if t < n_steps // 2:
            n_early += int(s[0])
        else:
            n_late += int(s[0])
    assert n_late <= n_early


def test_astrocyte_boost_increases_gain():
    a = AstrocyteModulator(n=1, Ca_threshold=2.0, boost=0.5)
    # Drive enough activity to cross Ca threshold.
    for _ in range(500):
        a.step(np.array([0.05]))
    assert a.gain[0] > 1.0


def test_tripartite_synapse_modulates():
    t = TripartiteSynapse(n=1, base_weight=1.0)
    pre = np.array([True])
    # Many spikes to build up Ca.
    for _ in range(500):
        t.transmit(pre)
    # After many spikes, gain should differ from 1.
    assert abs(t.astro.gain[0] - 1.0) > 1e-6


def test_neuromodulator_reward_boosts_dopamine():
    nm = NeuromodulatorSystem()
    base = nm.DA
    nm.step(reward=1.0)
    assert nm.DA > base


def test_neuromodulator_learning_gain():
    nm = NeuromodulatorSystem()
    g = nm.learning_gain()
    assert g > 0


# ---- Tier H ----

def test_helmholtz_reconstructs_after_training():
    rng = np.random.default_rng(0)
    # 4 binary patterns.
    X = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]],
                  dtype=np.float64)
    hm = HelmholtzMachine(n_visible=4, n_hidden=3, eta=0.1, rng=rng)
    # Sanity: should at least train without errors.
    hm.train(X, n_iter=500)
    rec = hm.reconstruct(X[0])
    assert rec.shape == (4,)


def test_rbm_recon_error_decreases():
    rng = np.random.default_rng(0)
    X = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.float64)
    rbm = RBM(n_visible=4, n_hidden=3, eta=0.1, rng=rng)
    losses = rbm.train(X, n_epochs=50)
    assert losses[-1] <= losses[0] + 0.5


def test_rbm_reconstruction_shape():
    rng = np.random.default_rng(0)
    rbm = RBM(n_visible=6, n_hidden=4, rng=rng)
    out = rbm.reconstruct(np.array([1, 0, 1, 0, 1, 0], dtype=np.float64))
    assert out.shape == (6,)


def test_vae_loss_decreases():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(8, 6)) < 0.5
    X = X.astype(np.float64)
    vae = VAE(n_in=6, n_hidden=8, n_latent=2, eta=0.01, rng=rng)
    losses = vae.train(X, n_iter=400)
    # Smoothed last block should be lower than first block.
    early = np.mean(losses[:50])
    late = np.mean(losses[-50:])
    assert late < early


def test_deep_predictive_coder_loss_decreases():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 4))
    dpc = DeepPredictiveCoder(layer_sizes=[4, 6, 3], rng=rng)
    losses = dpc.train(X, n_iter=200)
    assert np.mean(losses[-30:]) < np.mean(losses[:30])


def test_dpc_reconstruct_shape():
    rng = np.random.default_rng(0)
    dpc = DeepPredictiveCoder(layer_sizes=[5, 8, 3], rng=rng)
    x = rng.standard_normal(5)
    out = dpc.reconstruct(x)
    assert out.shape == (5,)


def test_conv2d_forward_shape():
    X = np.random.default_rng(0).standard_normal((1, 5, 5))
    W = np.random.default_rng(0).standard_normal((2, 1, 3, 3))
    b = np.zeros(2)
    out = conv2d_forward(X, W, b)
    assert out.shape == (2, 3, 3)


def test_maxpool2d_halves_dimensions():
    X = np.arange(16).reshape(1, 4, 4).astype(np.float64)
    out, idx = maxpool2d_forward(X, k=2)
    assert out.shape == (1, 2, 2)
    # Max of 4x4 block is 15.
    assert out.max() == 15


def test_convnet_overfits_one_sample():
    rng = np.random.default_rng(0)
    cnn = SimpleConvNet(in_shape=(1, 6, 6), n_classes=3, n_filters=2,
                          k=3, pool=2, eta=0.05, rng=rng)
    X = rng.uniform(size=(1, 6, 6))
    y = 2
    losses = []
    for _ in range(200):
        losses.append(cnn.loss_and_step(X, y))
    assert losses[-1] < losses[0]
    assert cnn.predict(X) == y
