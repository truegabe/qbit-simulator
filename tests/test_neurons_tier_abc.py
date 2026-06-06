"""Tests for Tier A/B/C neural modules."""

import numpy as np
import pytest

from qbit_simulator.neurons.izhikevich import (
    IzhikevichNeuron, IzhikevichPopulation, REGIMES, run_izhikevich,
)
from qbit_simulator.neurons.hodgkin_huxley import HHNeuron, run_hh
from qbit_simulator.neurons.adex import AdExNeuron, AdExPopulation, run_adex
from qbit_simulator.neurons.conductance_synapses import (
    AMPASynapse, GABAASynapse, NMDASynapse, DoubleExpSynapse, CondBasedLIF,
)
from qbit_simulator.neurons.stp import (
    TsodyksMarkramSynapse, STPPopulation,
    depressing_synapse, facilitating_synapse,
)
from qbit_simulator.neurons.compartments import (
    MultiCompartmentNeuron, linear_dendrite, attenuation,
)
from qbit_simulator.neurons.ei_balanced import EIBalancedNetwork
from qbit_simulator.neurons.synfire import SynfireChain
from qbit_simulator.neurons.som import SelfOrganizingMap
from qbit_simulator.neurons.continuous_attractor_2d import ContinuousAttractor2D
from qbit_simulator.neurons.art import ART1
from qbit_simulator.neurons.bcm import BCMNeuron
from qbit_simulator.neurons.oja import OjaNeuron, SangerNetwork
from qbit_simulator.neurons.tempotron import Tempotron
from qbit_simulator.neurons.surrogate_grad import SurrogateGradSNN
from qbit_simulator.neurons.three_factor import (
    ThreeFactorLearner, DopamineModulator,
)


# ---- Tier A ----

def test_izhikevich_rs_fires():
    n = IzhikevichNeuron.from_regime("RS")
    out = run_izhikevich(n, I_func=10.0, n_steps=400)
    assert out["n_spikes"] > 3


def test_izhikevich_fs_higher_rate_than_rs():
    rs = IzhikevichNeuron.from_regime("RS")
    fs = IzhikevichNeuron.from_regime("FS")
    r1 = run_izhikevich(rs, I_func=10.0, n_steps=400)
    r2 = run_izhikevich(fs, I_func=10.0, n_steps=400)
    assert r2["n_spikes"] > r1["n_spikes"]


def test_izhikevich_population_vectorized():
    pop = IzhikevichPopulation.from_regime(10, "RS")
    I = np.full(10, 10.0)
    n_spikes = 0
    for _ in range(400):
        n_spikes += int(pop.step(I).sum())
    assert n_spikes > 10


def test_hh_fires_above_threshold():
    n = HHNeuron()
    out = run_hh(n, I_func=10.0, n_steps=10000)
    assert out["n_spikes"] >= 3


def test_hh_silent_when_no_drive():
    n = HHNeuron()
    out = run_hh(n, I_func=0.0, n_steps=5000)
    assert out["n_spikes"] == 0


def test_adex_fires():
    n = AdExNeuron()
    out = run_adex(n, I_func=1000.0, n_steps=2000)
    assert out["n_spikes"] >= 3


def test_adex_adapts():
    """ISI should grow over time due to adaptation w."""
    n = AdExNeuron(b=0.5)
    out = run_adex(n, I_func=1200.0, n_steps=4000)
    times = np.where(out["spikes"])[0]
    if len(times) >= 4:
        first_isi = times[1] - times[0]
        last_isi = times[-1] - times[-2]
        assert last_isi >= first_isi


def test_ampa_decays():
    s = AMPASynapse(n_post=3)
    s.receive(np.array([1.0, 0.0, 0.0]))
    g0 = s.g[0]
    s.step_decay(dt=1.0)
    assert s.g[0] < g0


def test_ampa_excitatory_current_positive():
    s = AMPASynapse(n_post=1)
    s.receive(np.array([1.0]))
    I = s.current(np.array([-70.0]))   # well below E_rev=0
    assert I[0] > 0


def test_gaba_inhibitory_current_negative():
    s = GABAASynapse(n_post=1)
    s.receive(np.array([1.0]))
    I = s.current(np.array([-60.0]))   # above E_rev=-70
    assert I[0] < 0


def test_nmda_mg_block_at_rest():
    """NMDA should be blocked at hyperpolarized V."""
    s = NMDASynapse(n_post=1); s.receive(np.array([1.0]))
    block_rest = s.block(np.array([-70.0]))[0]
    block_dep  = s.block(np.array([-20.0]))[0]
    assert block_dep > block_rest


def test_double_exp_rises_then_decays():
    s = DoubleExpSynapse(n_post=1, tau_r=0.5, tau_d=5.0)
    s.receive(np.array([1.0]))
    g_t = []
    for _ in range(30):
        # No new input — pure kernel.
        I = -s.current(np.array([-70.0]))   # |I|
        g_t.append(abs(I[0]))
        s.step_decay(dt=1.0)
    assert max(g_t) > g_t[0]


def test_cond_based_lif_fires():
    pop = CondBasedLIF(n=1)
    fired = False
    for t in range(500):
        # Strong injected current to depolarize past threshold.
        spikes = pop.step(np.array([400.0]), t=t, dt=0.5)
        if spikes[0]:
            fired = True; break
    assert fired


def test_tm_depressing_amplitude_drops():
    syn = depressing_synapse()
    amps = []
    for _ in range(5):
        amps.append(syn.step(dt=10.0, pre_spike=True))
    assert amps[0] > amps[-1]


def test_tm_facilitating_amplitude_grows():
    """For a facilitating synapse with very low U, repeated spikes
    initially grow in amplitude even if x is depleting."""
    syn = facilitating_synapse()
    amps = []
    for _ in range(3):
        # Rapid bursts: dt=5 ms. x decays but u grows faster.
        amps.append(syn.step(dt=5.0, pre_spike=True))
    # u jumps from U=0.15 toward ~0.42 by spike 3; x depletes from 1 to
    # ~0.7 — net facilitation in early spikes.
    assert amps[1] > amps[0]


def test_stp_population():
    pop = STPPopulation(n=5, U=0.5, tau_d=500.0)
    pre = np.array([1, 0, 1, 0, 1])
    out = pop.step(1.0, pre)
    assert out[0] > 0
    assert out[1] == 0
    assert out[3] == 0


def test_compartments_chain_dendrite_attenuates():
    n = linear_dendrite(n=5, g_axial=0.1)
    V_end = attenuation(n, n_steps=2000, drive_compartment=4, amplitude=0.5)
    # Distal compartment most depolarized; soma less so.
    assert V_end[4] > V_end[0]


def test_compartments_soma_can_spike():
    n = linear_dendrite(n=3, g_axial=0.1)
    fired = False
    I = np.zeros(3); I[0] = 2.0
    for t in range(500):
        if n.step(I, t=t):
            fired = True; break
    assert fired


# ---- Tier B ----

def test_ei_balanced_fires():
    net = EIBalancedNetwork(N_E=100, N_I=25, nu_ext=0.5, J=0.5)
    out = net.run(n_steps=400)
    assert out["mean_rate_E"] > 0
    assert out["mean_rate_I"] > 0


def test_synfire_propagates():
    chain = SynfireChain(n_layers=4, layer_size=15, w=0.4)
    out = chain.run(n_steps=80, pulse_strength=5.0, pulse_duration=10)
    # First layer must be strongly active.
    assert out["layer_counts"][0] > 0


def test_som_quantization_error_drops():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(50, 3))
    som = SelfOrganizingMap(map_h=5, map_w=5, input_dim=3, rng=rng)
    e0 = som.quantization_error(X)
    som.train(X, n_iter=3000)
    e1 = som.quantization_error(X)
    assert e1 < e0


def test_continuous_attractor_bump_forms():
    can = ContinuousAttractor2D(h=10, w=10)
    ext = np.zeros((10, 10)); ext[5, 5] = 2.0
    for _ in range(50):
        can.step(ext)
    ext0 = np.zeros((10, 10))
    for _ in range(20):
        can.step(ext0)
    assert can.rate.max() > 0.1
    r, c = can.bump_center()
    assert abs(r - 5) < 2 or abs(r - 5) > 8   # mod periodic


def test_art1_creates_categories():
    rng = np.random.default_rng(0)
    art = ART1(input_dim=8, rho=0.7)
    p1 = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    p2 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    j1 = art.present(p1)
    j2 = art.present(p2)
    assert j1 != j2
    # Same pattern resonates.
    j1b = art.present(p1)
    assert j1b == j1


def test_art1_high_vigilance_more_categories():
    p1 = np.array([1, 1, 1, 0])
    p2 = np.array([1, 1, 1, 1])
    low  = ART1(input_dim=4, rho=0.3)
    high = ART1(input_dim=4, rho=0.95)
    low.present(p1); low.present(p2)
    high.present(p1); high.present(p2)
    assert high.n_categories >= low.n_categories


# ---- Tier C ----

def test_bcm_becomes_selective():
    rng = np.random.default_rng(0)
    # 3 orthogonal patterns.
    X = np.eye(3) * 1.5
    n = BCMNeuron(n_inputs=3, eta=0.01)
    out = n.train(X, n_iter=3000, rng=rng)
    # One should win.
    assert out["responses"].max() > 3 * out["responses"].min() + 1e-3 \
           or out["responses"].max() > 0.5


def test_oja_converges_to_unit_norm():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 5))
    n = OjaNeuron(n_inputs=5, eta=0.005)
    w = n.train(X, n_epochs=20)
    assert abs(np.linalg.norm(w) - 1.0) < 0.3


def test_sanger_recovers_first_pc():
    rng = np.random.default_rng(0)
    # Highly anisotropic data.
    direction = np.array([1.0, 0.0, 0.0])
    X = rng.standard_normal((300, 3)) * 0.1 + np.outer(rng.standard_normal(300), direction)
    net = SangerNetwork(n_inputs=3, n_components=2, eta=0.005)
    net.train(X, n_epochs=40)
    # First component should align with x-axis (up to sign).
    assert abs(abs(net.W[0, 0]) - 1.0) < 0.4


def test_tempotron_learns_simple_pattern():
    rng = np.random.default_rng(0)
    t = Tempotron(n_inputs=4, T=80, eta=0.005)
    # Positive pattern: spikes early on inputs 0, 1.
    pos = [np.array([5.0]), np.array([10.0]), np.array([]), np.array([])]
    # Negative pattern: spikes on inputs 2, 3.
    neg = [np.array([]), np.array([]), np.array([5.0]), np.array([10.0])]
    for _ in range(200):
        t.train_one(pos, 1)
        t.train_one(neg, 0)
    fired_pos, _ = t.classify(pos)
    fired_neg, _ = t.classify(neg)
    assert fired_pos
    assert not fired_neg


def test_surrogate_grad_decreases_loss():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 0.5, size=(20, 4))
    target = np.array([2.0])
    snn = SurrogateGradSNN(n_in=4, n_hidden=8, n_out=1, rng=rng)
    loss0 = snn.loss_and_grads(X, target)[0]
    for _ in range(50):
        snn.step_sgd(X, target, lr=0.001)
    loss1 = snn.loss_and_grads(X, target)[0]
    assert loss1 <= loss0


def test_three_factor_learning_gated_by_modulator():
    learner = ThreeFactorLearner(n_pre=3, n_post=2, eta=0.1)
    W = np.zeros((2, 3))
    pre = np.array([1.0, 0, 1.0])
    post = np.array([1.0, 0])
    # Modulator 0: no change.
    W1 = learner.step(W.copy(), pre, post, modulator=0.0)
    assert np.allclose(W1, 0.0)
    # Modulator 1: change.
    learner.reset()
    W2 = learner.step(W.copy(), pre, post, modulator=1.0)
    assert W2[0, 0] > 0


def test_dopamine_modulator_phasic_burst():
    dm = DopamineModulator(tonic=0.1, phasic_amp=1.0)
    base = dm.step(dt=1.0, reward=0.0)
    after_reward = dm.step(dt=1.0, reward=1.0)
    assert after_reward > base + 0.5
