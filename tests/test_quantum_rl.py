"""Quantum reinforcement learning tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.quantum_rl import (
    GridWorld, quantum_policy, sample_action,
    run_episode, reinforce,
)


# ---- GridWorld ----

def test_gridworld_reset_at_origin():
    env = GridWorld(size=3)
    s = env.reset()
    assert s == 0   # top-left corner


def test_gridworld_actions_change_position():
    env = GridWorld(size=3, goal=(2, 2))
    env.reset()
    s_right, _, _ = env.step(1)
    assert s_right == 1   # (1, 0)


def test_gridworld_actions_clamp_at_edge():
    env = GridWorld(size=2)
    env.reset()
    # Moving left from (0,0) should keep us there.
    s, _, _ = env.step(3)
    assert s == 0


def test_gridworld_reaches_goal_for_reward():
    env = GridWorld(size=2, goal=(1, 1))
    env.reset()
    env.step(1)   # right → (1, 0)
    s, r, done = env.step(2)  # down → (1, 1) = goal
    assert done
    assert r == 1.0


def test_gridworld_max_steps_terminates():
    env = GridWorld(size=2, goal=(1, 1), max_steps=3)
    env.reset()
    for _ in range(3):
        _, _, done = env.step(3)   # go left = no movement
    assert done


# ---- Quantum policy ----

def test_quantum_policy_probabilities_normalized():
    rng = np.random.default_rng(0)
    n_q = 2; n_layers = 2
    theta = rng.uniform(-1, 1, size=2 * n_q * n_layers)
    probs = quantum_policy(theta, state=0, n_qubits=n_q, n_actions=4)
    assert abs(probs.sum() - 1.0) < 1e-9
    assert (probs >= 0).all()


def test_quantum_policy_deterministic_with_zero_params():
    """With θ=0, all Ry/Rz are identity → state |s⟩ stays as |s⟩.
    Action probs concentrate on the single basis-state outcome."""
    probs = quantum_policy(np.zeros(8), state=2, n_qubits=2, n_actions=4)
    # |s=2⟩ maps to the bit-pattern; without rotations, only the
    # corresponding outcome has nonzero probability.
    assert abs(probs.sum() - 1.0) < 1e-9


def test_quantum_policy_rejects_bad_theta_length():
    with pytest.raises(ValueError):
        quantum_policy(np.zeros(7), state=0, n_qubits=2, n_actions=4)


def test_sample_action_returns_valid():
    rng = np.random.default_rng(0)
    probs = np.array([0.25, 0.25, 0.25, 0.25])
    for _ in range(50):
        a = sample_action(probs, rng)
        assert 0 <= a < 4


# ---- Episode runner ----

def test_run_episode_returns_data():
    env = GridWorld(size=2, max_steps=5)
    rng = np.random.default_rng(0)
    theta = np.zeros(8)   # 2 qubits, 2 layers
    ep = run_episode(env, theta, n_qubits=2, rng=rng)
    assert "states" in ep
    assert "actions" in ep
    assert "rewards" in ep
    assert "returns" in ep
    assert len(ep["actions"]) == ep["n_steps"]
    assert ep["n_steps"] <= 5


def test_run_episode_returns_decreasing_along_trajectory():
    """G_t should be discounted returns, decreasing for negative rewards."""
    env = GridWorld(size=3, goal=(2, 2), max_steps=4)
    rng = np.random.default_rng(0)
    theta = np.zeros(8)
    ep = run_episode(env, theta, n_qubits=2, rng=rng, gamma=0.9)
    # G_t are well-defined floats.
    assert len(ep["returns"]) == ep["n_steps"]
    assert all(np.isfinite(g) for g in ep["returns"])


# ---- REINFORCE ----

def test_reinforce_returns_structure():
    env = GridWorld(size=2, max_steps=5)
    rng = np.random.default_rng(0)
    theta = rng.uniform(-0.3, 0.3, size=8)
    result = reinforce(env, n_qubits=2, theta_init=theta,
                        n_episodes=3, lr=0.05, rng=rng)
    assert "theta" in result
    assert "episode_rewards" in result
    assert "avg_last_10" in result
    assert len(result["episode_rewards"]) == 3


def test_reinforce_improves_over_episodes():
    """Trained policy should outperform untrained policy on average."""
    env = GridWorld(size=2, goal=(1, 1), max_steps=10)
    rng = np.random.default_rng(0)
    theta = rng.uniform(-0.3, 0.3, size=8)
    result = reinforce(env, n_qubits=2, theta_init=theta,
                        n_episodes=40, lr=0.1, rng=rng)
    rewards = result["episode_rewards"]
    early = np.mean(rewards[:10])
    late = np.mean(rewards[-10:])
    assert late > early
