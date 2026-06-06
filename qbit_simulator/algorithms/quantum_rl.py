"""Quantum reinforcement learning: parameterized-circuit policies.

A quantum reinforcement-learning agent represents its policy
π(a | s; θ) by a parameterized quantum circuit. For state s ∈ {0,..,N-1}
and action a ∈ {0,..,K-1}:

    1. Encode state: prepare |s⟩.
    2. Apply parameterized ansatz U(θ).
    3. Measure: sample action a from |⟨a | U(θ) |s⟩|².

Train with classical policy-gradient (REINFORCE) on episodic rewards:

    ∇θ J(θ) = E_τ [ sum_t ∇θ log π(a_t | s_t; θ) · G_t ]

where G_t = sum_{t'>t} γ^{t'-t} r_{t'} is the return-to-go.

We demonstrate on a small **grid world** (2x2 or 3x3) — small enough
to encode each state as a basis state on log₂(N) qubits.

Provides:

  - `GridWorld(size, goal)`: discrete environment.
  - `quantum_policy(theta, state, n_qubits, n_actions)`: build the
    policy circuit + return action probabilities.
  - `sample_action(probs, rng)`: sample.
  - `reinforce(env, theta_init, n_episodes, lr)`: train.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


# ----------------------------------------------------------------------------
# Grid-world environment
# ----------------------------------------------------------------------------

@dataclass
class GridWorld:
    """Simple grid world: agent navigates a `size × size` grid to reach `goal`.

    Actions: 0=up, 1=right, 2=down, 3=left.
    Reward: -0.1 per step, +1 on reaching goal, episode ends on goal
    or after `max_steps`.
    """
    size: int = 2
    goal: tuple = (1, 1)
    max_steps: int = 20

    @property
    def n_states(self) -> int:
        return self.size * self.size

    @property
    def n_actions(self) -> int:
        return 4

    def reset(self) -> int:
        """Start at top-left corner; return state index."""
        self.x = 0
        self.y = 0
        self.steps = 0
        return self._state()

    def _state(self) -> int:
        return self.y * self.size + self.x

    def step(self, action: int) -> tuple[int, float, bool]:
        """Return (next_state, reward, done)."""
        if action == 0:    # up
            self.y = max(0, self.y - 1)
        elif action == 1:  # right
            self.x = min(self.size - 1, self.x + 1)
        elif action == 2:  # down
            self.y = min(self.size - 1, self.y + 1)
        elif action == 3:  # left
            self.x = max(0, self.x - 1)
        self.steps += 1
        done = (self.x, self.y) == self.goal or self.steps >= self.max_steps
        reward = 1.0 if (self.x, self.y) == self.goal else -0.1
        return self._state(), reward, done


# ----------------------------------------------------------------------------
# Quantum policy
# ----------------------------------------------------------------------------

# Pauli matrices
_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def _ry(theta: float) -> np.ndarray:
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _rz(theta: float) -> np.ndarray:
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)],
    ], dtype=np.complex128)


def _apply_1q(psi: np.ndarray, gate: np.ndarray, q: int, n: int) -> np.ndarray:
    """Apply 1q gate on qubit q (MSB-first: axis q = qubit q)."""
    shape = [2] * n
    psi_t = psi.reshape(shape)
    psi_t = np.moveaxis(psi_t, q, 0)
    psi_t = psi_t.reshape(2, -1)
    psi_t = gate @ psi_t
    psi_t = psi_t.reshape([2] + [2] * (n - 1))
    psi_t = np.moveaxis(psi_t, 0, q)
    return psi_t.reshape(2 ** n)


def _apply_cnot(psi: np.ndarray, control: int, target: int,
                 n: int) -> np.ndarray:
    new_psi = psi.copy()
    for idx in range(2 ** n):
        c_bit = (idx >> (n - 1 - control)) & 1
        if c_bit == 1:
            flipped = idx ^ (1 << (n - 1 - target))
            new_psi[idx] = psi[flipped]
    return new_psi


def quantum_policy(theta: np.ndarray, state: int, n_qubits: int,
                    n_actions: int = 4) -> np.ndarray:
    """Build the policy circuit and return action probabilities.

    Architecture: state-encoding layer (Ry rotations conditioned on bit
    pattern of `state`), then a hardware-efficient ansatz of Ry+Rz
    layers with nearest-neighbor CNOT entanglers. Output probabilities
    are obtained by marginalizing the measurement distribution to the
    first ⌈log₂(K)⌉ qubits.

    Args:
        theta:      parameter vector. Length = 2 · n_qubits · n_layers.
        state:      integer in [0, 2^n_qubits).
        n_qubits:   total qubits.
        n_actions:  number of actions.

    Returns:
        length-n_actions probability vector.
    """
    n_layers = len(theta) // (2 * n_qubits)
    if 2 * n_qubits * n_layers != len(theta):
        raise ValueError(
            f"theta length {len(theta)} != 2 * n_qubits * n_layers"
        )
    # Encode state as |state⟩ on n_qubits.
    psi = np.zeros(2 ** n_qubits, dtype=np.complex128)
    psi[state] = 1.0
    # Apply n_layers of (Ry, Rz on each qubit, then CNOT ladder).
    for L in range(n_layers):
        for q in range(n_qubits):
            theta_y = theta[L * 2 * n_qubits + q]
            theta_z = theta[L * 2 * n_qubits + n_qubits + q]
            psi = _apply_1q(psi, _ry(theta_y), q, n_qubits)
            psi = _apply_1q(psi, _rz(theta_z), q, n_qubits)
        if n_qubits > 1:
            for q in range(n_qubits - 1):
                psi = _apply_cnot(psi, q, q + 1, n_qubits)
    # Marginalize to the first ceil(log2(n_actions)) qubits.
    n_a_qubits = int(np.ceil(np.log2(n_actions)))
    probs = np.zeros(2 ** n_a_qubits)
    for idx in range(2 ** n_qubits):
        a = idx >> (n_qubits - n_a_qubits)
        probs[a] += abs(psi[idx]) ** 2
    return probs[:n_actions] / probs[:n_actions].sum()


def sample_action(probs: np.ndarray,
                   rng: np.random.Generator) -> int:
    return int(rng.choice(len(probs), p=probs))


# ----------------------------------------------------------------------------
# REINFORCE training
# ----------------------------------------------------------------------------

def run_episode(env: GridWorld, theta: np.ndarray, n_qubits: int,
                  rng: np.random.Generator,
                  gamma: float = 0.95) -> dict:
    """Run one episode following the quantum policy. Returns the
    transition data needed for REINFORCE."""
    state = env.reset()
    states, actions, rewards = [], [], []
    done = False
    while not done:
        probs = quantum_policy(theta, state, n_qubits, env.n_actions)
        a = sample_action(probs, rng)
        states.append(state)
        actions.append(a)
        state, r, done = env.step(a)
        rewards.append(r)
    # Compute returns G_t.
    returns = np.zeros(len(rewards))
    G = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        G = rewards[t] + gamma * G
        returns[t] = G
    return {
        "states":     states,
        "actions":    actions,
        "rewards":    rewards,
        "returns":    returns,
        "n_steps":    len(rewards),
        "total_reward": sum(rewards),
    }


def reinforce(
    env: GridWorld,
    n_qubits: int,
    theta_init: np.ndarray,
    n_episodes: int = 100,
    lr: float = 0.05,
    gamma: float = 0.95,
    rng: np.random.Generator | None = None,
) -> dict:
    """Train the quantum policy with REINFORCE."""
    rng = rng or np.random.default_rng()
    theta = theta_init.copy()
    episode_rewards = []
    for ep in range(n_episodes):
        episode = run_episode(env, theta, n_qubits, rng, gamma=gamma)
        episode_rewards.append(episode["total_reward"])
        # Compute gradient: sum_t G_t · ∇log π(a_t | s_t).
        # Use finite-difference for the log-prob gradient.
        grad = np.zeros_like(theta)
        eps = np.pi / 6   # parameter-shift-like
        for k in range(len(theta)):
            theta_p = theta.copy(); theta_p[k] += eps
            theta_m = theta.copy(); theta_m[k] -= eps
            for t_step in range(episode["n_steps"]):
                s = episode["states"][t_step]
                a = episode["actions"][t_step]
                p_plus = quantum_policy(theta_p, s, n_qubits, env.n_actions)[a]
                p_minus = quantum_policy(theta_m, s, n_qubits, env.n_actions)[a]
                p_curr = quantum_policy(theta, s, n_qubits, env.n_actions)[a]
                # d/dθ log p ≈ (log(p+) - log(p-)) / (2eps)
                if p_plus > 1e-9 and p_minus > 1e-9:
                    grad[k] += (np.log(p_plus) - np.log(p_minus)) / (2 * eps) \
                                * episode["returns"][t_step]
        theta = theta + lr * grad
    return {
        "theta":              theta,
        "episode_rewards":    episode_rewards,
        "avg_last_10":        float(np.mean(episode_rewards[-10:])),
    }
