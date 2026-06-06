"""Tests for Tier I/J/K modules: temporal, RL, brain regions."""

import numpy as np
import pytest

from qbit_simulator.neurons.vanilla_rnn import VanillaRNN
from qbit_simulator.neurons.lstm import LSTMCell
from qbit_simulator.neurons.gru import GRUCell
from qbit_simulator.neurons.transformer import (
    MultiHeadAttention, TransformerBlock, positional_encoding,
    scaled_dot_product_attention,
)
from qbit_simulator.neurons.force_learning import FORCE
from qbit_simulator.neurons.htm import SpatialPooler, TemporalMemory, HTM
from qbit_simulator.neurons.q_learning import (
    QLearning, SARSA, ExpectedSARSA, run_episode_grid,
)
from qbit_simulator.neurons.actor_critic import REINFORCE, ActorCritic
from qbit_simulator.neurons.successor_representation import SuccessorRepresentation
from qbit_simulator.neurons.dyna_q import DynaQ
from qbit_simulator.neurons.options import Option, OptionsAgent
from qbit_simulator.neurons.hippocampus import Hippocampus
from qbit_simulator.neurons.entorhinal_grid_module import GridModule, GridSystem
from qbit_simulator.neurons.pfc_working_memory import PFCWorkingMemory
from qbit_simulator.neurons.amygdala import Amygdala
from qbit_simulator.neurons.olfactory_bulb import OlfactoryBulb
from qbit_simulator.neurons.thalamocortical import Thalamocortical


# ---- Tier I: temporal sequences ----

def test_rnn_loss_decreases():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 3))
    Y = np.tanh(X[:, :1])
    rnn = VanillaRNN(n_in=3, n_hidden=8, n_out=1, eta=0.05, rng=rng)
    loss0 = rnn.loss_and_grads(X, Y)[0]
    for _ in range(100):
        rnn.step_sgd(X, Y)
    loss1 = rnn.loss_and_grads(X, Y)[0]
    assert loss1 < loss0


def test_lstm_loss_decreases():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 3))
    Y = np.sin(X[:, :1])
    lstm = LSTMCell(n_in=3, n_hidden=6, n_out=1, eta=0.05, rng=rng)
    loss0 = lstm.loss_and_grads(X, Y)[0]
    for _ in range(100):
        lstm.step_sgd(X, Y)
    loss1 = lstm.loss_and_grads(X, Y)[0]
    assert loss1 < loss0


def test_gru_loss_decreases():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 3))
    Y = np.sin(X[:, :1])
    gru = GRUCell(n_in=3, n_hidden=6, n_out=1, eta=0.05, rng=rng)
    loss0 = gru.loss_and_step(X, Y)
    for _ in range(50):
        gru.loss_and_step(X, Y)
    loss1 = gru.loss_and_step(X, Y)
    assert loss1 <= loss0 + 0.5


def test_attention_self_attention_runs():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 8))
    mha = MultiHeadAttention(d_model=8, n_heads=2, rng=rng)
    out = mha.forward(X)
    assert out["output"].shape == (5, 8)
    # Attention rows sum to 1.
    A = out["attention"][0]
    assert np.allclose(A.sum(axis=-1), 1.0)


def test_scaled_dot_product_attention_basic():
    Q = np.eye(3); K = np.eye(3); V = np.eye(3) * 2
    out, A = scaled_dot_product_attention(Q, K, V)
    assert out.shape == (3, 3)
    assert np.allclose(A.sum(axis=-1), 1.0)


def test_transformer_block_preserves_shape():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 8))
    blk = TransformerBlock(d_model=8, n_heads=2, rng=rng)
    out = blk.forward(X)
    assert out.shape == (4, 8)


def test_positional_encoding_shape():
    pe = positional_encoding(T=10, d_model=8)
    assert pe.shape == (10, 8)


def test_force_learning_runs():
    rng = np.random.default_rng(0)
    f = FORCE(n=50, rng=rng)
    out = f.run(lambda t: np.sin(0.1 * t), n_steps=500, train=True)
    assert out.shape == (500,)


def test_htm_pipeline_runs():
    htm = HTM(n_input=20, n_columns=64, cells_per_column=4, sparsity=0.1)
    x = np.zeros(20); x[:5] = 1
    out = htm.step(x)
    assert "active" in out and "predictive" in out


def test_spatial_pooler_produces_sparse_output():
    sp = SpatialPooler(n_input=30, n_columns=100, sparsity=0.1)
    x = np.random.default_rng(0).uniform(size=30) > 0.5
    a = sp.compute(x.astype(float))
    # ~ 10 active columns. Slack for ties at the top-k threshold.
    assert 5 <= a.sum() <= 30


# ---- Tier J: RL agents ----

def test_q_learning_solves_chain():
    agent = QLearning(n_states=4, n_actions=2, alpha=0.3, eps=0.5)
    for _ in range(500):
        run_episode_grid(agent, n_states=4, max_steps=30)
    # Should pick right (action 1) in early states.
    assert agent.Q[0, 1] > agent.Q[0, 0]


def test_sarsa_solves_chain():
    agent = SARSA(n_states=4, n_actions=2, alpha=0.3, eps=0.5)
    for _ in range(500):
        run_episode_grid(agent, n_states=4, max_steps=30)
    assert agent.Q[0, 1] > agent.Q[0, 0]


def test_expected_sarsa_updates():
    agent = ExpectedSARSA(n_states=3, n_actions=2)
    delta = agent.update(0, 0, 1.0, 1, done=True)
    assert agent.Q[0, 0] > 0
    assert delta > 0


def test_reinforce_increases_action_prob():
    agent = REINFORCE(n_states=2, n_actions=2, alpha=0.2)
    # Train: in state 0, action 1 always gets reward.
    for _ in range(50):
        traj = [(0, 1, 1.0)]
        agent.update(traj)
    p = agent.policy(0)
    assert p[1] > p[0]


def test_actor_critic_value_grows():
    agent = ActorCritic(n_states=3, n_actions=2, alpha=0.2, beta=0.2)
    for _ in range(50):
        agent.step_update(0, 1, 1.0, 1, done=True)
    assert agent.V[0] > 0


def test_successor_representation_learns_chain():
    sr = SuccessorRepresentation(n_states=5, alpha=0.3)
    # Walk 0→1→2→3→4 repeatedly.
    for _ in range(50):
        for s in range(4):
            r = 1.0 if s + 1 == 4 else 0.0
            sr.update(s, s + 1, r)
    # State 0 should have positive SR for state 4.
    assert sr.M[0, 4] > 0


def test_dyna_q_planning():
    agent = DynaQ(n_states=4, n_actions=2, n_plan=5)
    # Simulate transitions.
    for _ in range(50):
        agent.step(0, 1, 1.0, 1, done=True)
    assert agent.Q[0, 1] > 0


def test_options_run_one_option():
    options = [
        Option("forward", initiation={0, 1}, policy={0: 1, 1: 1},
               termination={2}),
    ]
    agent = OptionsAgent(n_states=3, options=options)

    def env(s, a):
        s_next = min(s + 1, 2)
        return s_next, (1.0 if s_next == 2 else 0.0), s_next == 2

    s_term, cum, k = agent.run_option(0, 0, env)
    assert s_term == 2 and k == 2


# ---- Tier K: brain regions ----

def test_hippocampus_can_recall_stored_pattern():
    rng = np.random.default_rng(0)
    h = Hippocampus(n_input=20, n_dg=64, n_ca3=32, sparsity=0.1, rng=rng)
    p = (rng.uniform(size=20) > 0.5).astype(float)
    h.store(p)
    # Retrieve from the same cue.
    r = h.retrieve(p)
    assert r.sum() > 0


def test_grid_module_periodic_response():
    gm = GridModule(n_cells=8, scale=1.0)
    pos1 = np.array([0.0, 0.0])
    pos2 = np.array([1.0, 0.0])     # one full period away
    r1 = gm.firing(pos1)
    r2 = gm.firing(pos2)
    # Responses should be roughly correlated (periodicity).
    corr = np.corrcoef(r1, r2)[0, 1]
    assert not np.isnan(corr)


def test_grid_system_multi_scale():
    gs = GridSystem(scales=[1.0, 2.0], n_cells_per_module=4)
    r = gs.firing(np.array([0.3, 0.4]))
    assert r.shape == (8,)


def test_pfc_working_memory_store_and_read():
    pfc = PFCWorkingMemory(n_slots=2, n_features=4)
    x = np.array([1.0, 0.5, -0.5, 0.3])
    slot = pfc.store(x, gate=1.0)
    assert slot == 0
    r = pfc.read(0)
    assert np.allclose(r, x)


def test_pfc_decay_eventually_empties():
    pfc = PFCWorkingMemory(n_slots=1, decay=0.5, threshold=0.5)
    pfc.store(np.ones(8))
    for _ in range(5):
        pfc.step()
    # Should have decayed below threshold.
    assert pfc.read(0) is None


def test_amygdala_acquires_fear():
    amy = Amygdala(n_stimuli=2, alpha=0.3)
    cs = np.array([1.0, 0.0])
    for _ in range(20):
        amy.trial(cs, us=1.0)
    fear = amy.fear_response(cs)
    assert fear > 0.5


def test_amygdala_extinction_reduces_fear():
    amy = Amygdala(n_stimuli=2, alpha=0.3)
    cs = np.array([1.0, 0.0])
    for _ in range(20):
        amy.trial(cs, us=1.0)
    after_acq = amy.fear_response(cs)
    for _ in range(30):
        amy.trial(cs, us=0.0)
    after_ext = amy.fear_response(cs)
    assert after_ext < after_acq


def test_olfactory_bulb_runs():
    ob = OlfactoryBulb(n_mitral=10, n_granule=20)
    odor = np.zeros(10); odor[:3] = 1.5
    out = ob.run(odor, n_steps=50)
    assert out["mitral"].shape == (50, 10)


def test_thalamocortical_runs():
    tc = Thalamocortical(n_channels=4)
    sensory = np.array([1.0, 0.5, 0.2, 0.0])
    out = tc.run(sensory, n_steps=80)
    assert out["cortex"].shape == (80, 4)
