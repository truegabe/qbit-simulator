"""Measurement: collapse a state vector by random sampling."""

from __future__ import annotations

import numpy as np


def sample(probabilities: np.ndarray, shots: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    p = np.asarray(probabilities, dtype=np.float64)
    # Clip tiny negative values from floating-point drift, then renormalize.
    p = np.clip(p, 0.0, None)
    total = p.sum()
    if total <= 0.0:
        raise ValueError("Probabilities sum to zero; cannot sample.")
    p = p / total
    return rng.choice(len(p), size=shots, p=p)


def measure(state: np.ndarray, rng: np.random.Generator | None = None) -> tuple[int, np.ndarray]:
    """Single-shot measurement in the computational basis.

    Returns (outcome_index, collapsed_state).
    """
    probs = np.abs(state) ** 2
    outcome = int(sample(probs, shots=1, rng=rng)[0])
    collapsed = np.zeros_like(state)
    collapsed[outcome] = 1.0
    return outcome, collapsed
