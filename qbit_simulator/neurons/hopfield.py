"""Classical Hopfield network — associative (content-addressable) memory.

A Hopfield network (Hopfield 1982) is a recurrent binary neural network
that stores patterns as the stable fixed points (attractors) of its
dynamics. Given a NOISY or PARTIAL version of a stored pattern, the
network "completes" / "cleans up" the pattern by iteratively updating
neurons toward lower energy.

Architecture:
  - N binary neurons with states s_i ∈ {-1, +1}.
  - Symmetric weight matrix W_{ij} (W_{ii} = 0, W_{ij} = W_{ji}).
  - Update rule (asynchronous / Glauber):
        s_i ← sign( Σ_j W_{ij} s_j  +  θ_i )

  - Energy function:
        E(s)  =  -1/2 · Σ_{i,j} W_{ij} s_i s_j  -  Σ_i θ_i s_i

    Each update step decreases E (or leaves it equal); convergence to
    a fixed point is guaranteed.

Hebbian storage rule for P patterns {ξ^μ}_{μ=1..P}:

        W_{ij}  =  (1/N) · Σ_μ ξ_i^μ · ξ_j^μ              (i ≠ j)
        W_{ii}  =  0

Capacity: ~0.14 N patterns can be reliably stored (Amit-Gutfreund-
Sompolinsky 1985).

This module provides:

  - `HopfieldNetwork(n)`: the network with weights + state.
  - `.store_patterns(patterns)`: Hebbian + zero-diagonal weight setup.
  - `.energy(state)`: compute E(s).
  - `.update(state, mode="async"|"sync")`: one update sweep.
  - `.retrieve(probe, max_iter)`: iterate until convergence; return
    the recovered pattern.
  - `pattern_overlap(a, b)`: Hamming-style cosine similarity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class HopfieldNetwork:
    """A classical Hopfield network on N binary (±1) neurons."""
    n: int
    weights: np.ndarray = field(default=None, repr=False)
    thresholds: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = np.zeros((self.n, self.n), dtype=np.float64)
        if self.thresholds is None:
            self.thresholds = np.zeros(self.n, dtype=np.float64)
        if self.weights.shape != (self.n, self.n):
            raise ValueError(
                f"weights shape {self.weights.shape} != ({self.n}, {self.n})"
            )

    # ---- Pattern storage ----

    def store_patterns(self, patterns: np.ndarray) -> None:
        """Hebbian storage of P patterns.

        Args:
            patterns: shape (P, N), entries in {-1, +1}.
        """
        if patterns.ndim != 2:
            raise ValueError("patterns must be a 2D array (P, N)")
        if patterns.shape[1] != self.n:
            raise ValueError(
                f"pattern length {patterns.shape[1]} != network size {self.n}"
            )
        unique_values = set(np.unique(patterns).tolist())
        if not unique_values.issubset({-1, 1, -1.0, 1.0}):
            raise ValueError(
                "patterns must be ±1 only; got " + str(unique_values)
            )
        P = patterns.shape[0]
        # W_{ij} = (1/N) Σ_μ ξ_i^μ · ξ_j^μ.
        W = (patterns.T @ patterns).astype(np.float64) / self.n
        np.fill_diagonal(W, 0.0)
        self.weights = W

    def add_pattern(self, pattern: np.ndarray) -> None:
        """Incrementally add ONE pattern (cheap online learning)."""
        if pattern.shape != (self.n,):
            raise ValueError(f"pattern shape {pattern.shape} != ({self.n},)")
        if not set(np.unique(pattern).tolist()).issubset({-1, 1, -1.0, 1.0}):
            raise ValueError("pattern entries must be ±1")
        outer = np.outer(pattern, pattern).astype(np.float64) / self.n
        np.fill_diagonal(outer, 0.0)
        self.weights = self.weights + outer

    # ---- Energy ----

    def energy(self, state: np.ndarray) -> float:
        """E(s) = -1/2 · s^T W s − θ^T s."""
        s = np.asarray(state, dtype=np.float64)
        return float(-0.5 * s @ self.weights @ s - self.thresholds @ s)

    # ---- Dynamics ----

    def update_async(self, state: np.ndarray,
                       rng: np.random.Generator | None = None) -> np.ndarray:
        """One sweep of asynchronous (sequential) updates in random order.

        Each neuron updates to sign(W · s + θ); since updates are
        sequential, intermediate state changes propagate within one sweep.
        """
        rng = rng or np.random.default_rng()
        s = np.asarray(state, dtype=np.float64).copy()
        order = rng.permutation(self.n)
        for i in order:
            h_i = self.weights[i] @ s + self.thresholds[i]
            s[i] = 1.0 if h_i >= 0 else -1.0
        return s

    def update_sync(self, state: np.ndarray) -> np.ndarray:
        """One synchronous update: all neurons computed from the same s.

        Can oscillate between two configurations (Glauber's caveat).
        """
        s = np.asarray(state, dtype=np.float64)
        h = self.weights @ s + self.thresholds
        return np.where(h >= 0, 1.0, -1.0)

    def retrieve(
        self, probe: np.ndarray, max_iter: int = 100,
        mode: str = "async",
        rng: np.random.Generator | None = None,
    ) -> dict:
        """Iterate updates from `probe` until a fixed point is reached.

        Args:
            probe:    starting (possibly corrupted) pattern.
            max_iter: maximum sweeps.
            mode:     "async" (Glauber, always converges) or "sync"
                      (may oscillate).
            rng:      generator for async-update order.

        Returns:
            dict with retrieved_state, n_iter, converged, energy.
        """
        s = np.asarray(probe, dtype=np.float64).copy()
        E_prev = self.energy(s)
        history = [s.copy()]
        for it in range(max_iter):
            if mode == "async":
                s_new = self.update_async(s, rng=rng)
            elif mode == "sync":
                s_new = self.update_sync(s)
            else:
                raise ValueError(f"unknown mode {mode}")
            E_new = self.energy(s_new)
            history.append(s_new.copy())
            if np.array_equal(s_new, s):
                return {
                    "retrieved_state": s_new,
                    "n_iter":          it + 1,
                    "converged":       True,
                    "energy":          E_new,
                    "history":         history,
                }
            s = s_new
            E_prev = E_new
        return {
            "retrieved_state": s,
            "n_iter":          max_iter,
            "converged":       False,
            "energy":          self.energy(s),
            "history":         history,
        }


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def pattern_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized overlap (cosine similarity) of two ±1 patterns:
        m(a, b) = (1/N) · a · b   ∈ [-1, +1]
    1.0 = identical, -1.0 = perfectly anti-correlated, 0.0 = orthogonal.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("shape mismatch")
    return float((a @ b) / a.size)


def corrupt_pattern(pattern: np.ndarray, p_flip: float,
                      rng: np.random.Generator) -> np.ndarray:
    """Flip each bit of `pattern` independently with probability p_flip.
    Returns a new array; original is unchanged."""
    p = pattern.copy()
    flip = rng.uniform(size=p.shape) < p_flip
    p[flip] *= -1
    return p


def random_pattern(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a uniformly-random ±1 pattern."""
    return rng.choice([-1, 1], size=n).astype(np.float64)


# ----------------------------------------------------------------------------
# Capacity estimation
# ----------------------------------------------------------------------------

def estimate_capacity(n: int, n_patterns_list: list[int],
                        p_flip: float = 0.1, n_trials: int = 20,
                        rng: np.random.Generator | None = None) -> dict:
    """Empirically test how many random patterns a Hopfield network of
    size N can reliably retrieve under `p_flip` corruption.

    The asymptotic result: P_max ≈ 0.14 · N.

    Returns:
        dict mapping P → retrieval success rate.
    """
    rng = rng or np.random.default_rng()
    results: dict[int, float] = {}
    for P in n_patterns_list:
        net = HopfieldNetwork(n=n)
        patterns = np.array(
            [random_pattern(n, rng) for _ in range(P)]
        )
        net.store_patterns(patterns)
        successes = 0
        for trial in range(n_trials):
            idx = int(rng.integers(0, P))
            probe = corrupt_pattern(patterns[idx], p_flip, rng)
            r = net.retrieve(probe, max_iter=30, rng=rng)
            overlap = pattern_overlap(r["retrieved_state"], patterns[idx])
            if overlap > 0.95:
                successes += 1
        results[P] = successes / n_trials
    return results
