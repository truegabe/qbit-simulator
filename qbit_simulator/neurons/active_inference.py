"""Active inference: predictive coding extended with action.

Karl Friston's active inference framework extends predictive coding by
treating actions as another way to reduce prediction error. Where
classical predictive coding only adjusts internal beliefs to match
sensations, active inference also adjusts the WORLD (via motor actions)
to match predictions.

Mathematically: the agent minimizes Expected Free Energy

    G(π) = E_q[ log q(s | π) − log p(o, s | π) ]

over candidate POLICIES π (sequences of actions). The policy that
minimizes G is selected.

For tractable computation, this module implements a simplified version
on a small grid world:

  - `Agent(env_size, n_actions)`: the active-inference agent.
  - `.set_preference(preferred_obs)`: tells the agent what sensory
    state it wants (priors over observations).
  - `.step(observation)`: receive obs, update belief about world state,
    pick best action by simulating policies and computing EFE.
  - `GridWorld`: a tiny "outside world" the agent acts in.

The agent uses a learned forward model B[s, s', a] = P(s' | s, a)
plus a likelihood A[o, s] = P(o | s). For simplicity we provide both
as user-supplied tensors (or random).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------------
# Toy grid world
# ----------------------------------------------------------------------------

@dataclass
class GridWorld:
    """1D corridor with `n_states` cells. Actions: 0=left, 1=stay, 2=right.

    Observation = state (fully observable for simplicity).
    """
    n_states: int = 5
    state:    int = 0
    goal:     int = 4

    def reset(self, start: int = 0) -> int:
        self.state = start
        return self.state

    def step(self, action: int) -> tuple[int, float]:
        if action == 0:
            self.state = max(0, self.state - 1)
        elif action == 2:
            self.state = min(self.n_states - 1, self.state + 1)
        # action == 1: stay.
        reward = 1.0 if self.state == self.goal else 0.0
        return self.state, reward


# ----------------------------------------------------------------------------
# Active-inference agent
# ----------------------------------------------------------------------------

@dataclass
class ActiveInferenceAgent:
    """Discrete-state active-inference agent.

    Internal model:
      - A: likelihood matrix, shape (n_obs, n_states). A[o, s] = P(o|s).
      - B: transition tensor, shape (n_states, n_states, n_actions).
           B[s', s, a] = P(s' | s, a).
      - C: log preferences over observations, shape (n_obs,).
           Higher = more preferred.

    Belief state: posterior over states q(s).
    """
    n_states:  int
    n_actions: int = 3
    n_obs:     int | None = None
    A:         np.ndarray = field(default=None, repr=False)
    B:         np.ndarray = field(default=None, repr=False)
    C:         np.ndarray = field(default=None, repr=False)
    belief:    np.ndarray = field(default=None, repr=False)
    policy_depth: int = 2

    def __post_init__(self) -> None:
        if self.n_obs is None:
            self.n_obs = self.n_states
        if self.A is None:
            # Perfectly informative likelihood by default.
            self.A = np.eye(self.n_obs, self.n_states)
        if self.B is None:
            # Default transitions: 0=left, 1=stay, 2=right (deterministic
            # within bounds).
            self.B = np.zeros((self.n_states, self.n_states, self.n_actions))
            for s in range(self.n_states):
                self.B[max(s - 1, 0),                s, 0] = 1.0  # left
                self.B[s,                            s, 1] = 1.0  # stay
                self.B[min(s + 1, self.n_states-1),  s, 2] = 1.0  # right
        if self.C is None:
            self.C = np.zeros(self.n_obs)
        if self.belief is None:
            self.belief = np.ones(self.n_states) / self.n_states

    # ---- preferences ----

    def set_preference(self, preferred_obs: int,
                          weight: float = 4.0,
                          gradient: bool = True) -> None:
        """Set preferences over observations.

        If `gradient=True` (default), states closer to `preferred_obs`
        also get partial preference — short-horizon planning can then
        still navigate toward the goal.
        """
        if gradient:
            distances = np.abs(np.arange(self.n_obs) - preferred_obs)
            max_d = max(distances.max(), 1)
            self.C = weight * (1.0 - distances / max_d)
        else:
            self.C = np.zeros(self.n_obs)
            self.C[preferred_obs] = weight

    # ---- inference: update belief from new observation ----

    def update_belief(self, observation: int) -> np.ndarray:
        """Bayesian belief update: q(s) ∝ A[o, s] · q_prev(s)."""
        likelihood = self.A[observation, :]
        post = likelihood * self.belief
        if post.sum() < 1e-12:
            post = np.ones(self.n_states) / self.n_states
        else:
            post = post / post.sum()
        self.belief = post
        return post

    # ---- policy evaluation: Expected Free Energy ----

    def predict_belief_under_action(
        self, q: np.ndarray, action: int,
    ) -> np.ndarray:
        """One-step state belief prediction."""
        return self.B[:, :, action] @ q

    def predict_observation(self, q: np.ndarray) -> np.ndarray:
        """Marginal P(o) under current belief: P(o) = sum_s A[o, s] q(s)."""
        return self.A @ q

    def expected_free_energy(self, policy: list[int]) -> float:
        """G(π) = − sum_t C^T P_t(o)  +  (small ambiguity term).

        Simplified: only the "pragmatic value" term (preference matching).
        Negative because we want to MINIMIZE G — but for convenience we
        compute the score so HIGHER score = lower G.
        """
        q = self.belief.copy()
        score = 0.0
        for a in policy:
            q = self.predict_belief_under_action(q, a)
            p_o = self.predict_observation(q)
            score += float(self.C @ p_o)
        # Negative entropy (uncertainty penalty), small contribution.
        # We omit it for simplicity.
        return score   # higher is better

    def best_action(self) -> int:
        """Pick the FIRST action of the best policy of depth `policy_depth`."""
        from itertools import product
        best_score = -np.inf
        best_first = 0
        for policy in product(range(self.n_actions), repeat=self.policy_depth):
            score = self.expected_free_energy(list(policy))
            if score > best_score:
                best_score = score
                best_first = policy[0]
        return best_first

    # ---- one full step in the world ----

    def step(self, observation: int) -> int:
        """Receive an observation, update belief, return the action to take."""
        self.update_belief(observation)
        return self.best_action()


# ----------------------------------------------------------------------------
# Episode driver
# ----------------------------------------------------------------------------

def run_episode(
    env: GridWorld, agent: ActiveInferenceAgent,
    max_steps: int = 20, start_state: int = 0,
) -> dict:
    """Run one episode: agent acts in env until goal reached or max steps.

    Returns:
        dict with state_trajectory, action_trajectory, total_reward,
        reached_goal.
    """
    s = env.reset(start_state)
    obs = s   # fully observable
    states = [s]
    actions = []
    total_reward = 0.0
    for t in range(max_steps):
        a = agent.step(obs)
        s, r = env.step(a)
        actions.append(a)
        states.append(s)
        total_reward += r
        obs = s
        if s == env.goal:
            return {
                "states":        states,
                "actions":       actions,
                "total_reward":  total_reward,
                "reached_goal":  True,
                "n_steps":       t + 1,
            }
    return {
        "states":        states,
        "actions":       actions,
        "total_reward":  total_reward,
        "reached_goal":  False,
        "n_steps":       max_steps,
    }
