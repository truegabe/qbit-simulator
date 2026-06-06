"""Episodic memory: sparse distributed storage indexed by context.

The hippocampus encodes EPISODIC memories — specific events tied to a
context (time, place, surrounding stimuli). The canonical model is
Kanerva's Sparse Distributed Memory (Kanerva 1988):

    Storage:  given (key, value), find the K nearest "address" hash
              locations to `key` and write `value` to each.
    Recall:   given a `query` key, find the K nearest addresses and
              READ the average value stored there.

Robust to:
  - Noisy queries (returns nearest match).
  - Partial keys (cue-completion).
  - Multiple competing memories per address (interference is bounded
    by sparsity).

This module also provides a more biologically-flavored version using
RANDOM BINARY hashing and dot-product similarity.

Provides:

  - `SparseDistributedMemory(address_dim, n_addresses, k_nearest)`:
    Kanerva-style associative memory.
  - `.write(key, value)`, `.read(query)`.
  - `.recall_with_partial_cue(partial_key)`: graceful degradation.
  - `HippocampalMemory(context_dim, content_dim)`: a higher-level
    interface that binds context to content (e.g. "at location X I saw Y").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------------
# Sparse Distributed Memory (Kanerva)
# ----------------------------------------------------------------------------

@dataclass
class SparseDistributedMemory:
    """SDM with bipolar (±1) addresses and content vectors.

    Internal storage:
      - `addresses`:  shape (n_addresses, address_dim), random ±1 hash
                      locations.
      - `counters`:   shape (n_addresses, content_dim), integer counts.

    Writing `value` to address row a adds `value` to counters[a].
    Reading takes the sign of the accumulated counters at the K rows
    closest to the query.
    """
    address_dim: int = 64
    content_dim: int = 64
    n_addresses: int = 200
    k_nearest:   int = 10
    seed:        int = 0

    addresses: np.ndarray = field(default=None, repr=False)
    counters:  np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        if self.addresses is None:
            self.addresses = rng.choice([-1, 1], size=(self.n_addresses,
                                                         self.address_dim))
        if self.counters is None:
            self.counters = np.zeros((self.n_addresses, self.content_dim),
                                       dtype=np.float64)

    # ---- internal ----

    def _nearest_indices(self, key: np.ndarray) -> np.ndarray:
        """Return the indices of the K addresses closest to `key`
        (by Hamming-like negative dot-product on ±1 vectors)."""
        if key.shape != (self.address_dim,):
            raise ValueError(
                f"key dim {key.shape} != ({self.address_dim},)"
            )
        # Higher dot product = closer (for ±1 vectors).
        dots = self.addresses @ key
        return np.argsort(-dots)[:self.k_nearest]

    # ---- public API ----

    def write(self, key: np.ndarray, value: np.ndarray) -> None:
        """Store (key, value) at the K nearest hash locations."""
        if value.shape != (self.content_dim,):
            raise ValueError(
                f"value dim {value.shape} != ({self.content_dim},)"
            )
        for idx in self._nearest_indices(key):
            self.counters[idx] += value

    def read(self, query: np.ndarray) -> np.ndarray:
        """Recall: average counters at the K nearest hash locations,
        return sign as a ±1 content vector."""
        relevant = self.counters[self._nearest_indices(query)]
        avg = relevant.mean(axis=0)
        # Bipolar output.
        return np.where(avg > 0, 1.0, np.where(avg < 0, -1.0, 0.0))

    def read_continuous(self, query: np.ndarray) -> np.ndarray:
        """Raw averaged counters (before sign quantization)."""
        relevant = self.counters[self._nearest_indices(query)]
        return relevant.mean(axis=0)

    def recall_with_partial_cue(
        self, partial_key: np.ndarray, mask: np.ndarray,
    ) -> np.ndarray:
        """Recall when only SOME bits of the key are known.

        Args:
            partial_key: the partial key (entries at unknown positions
                         are ignored).
            mask:        bool array of same shape as key, True where the
                         partial key is known.

        Strategy: pad unknown bits with 0 (so they don't bias the dot
        product), find nearest addresses, read.
        """
        if partial_key.shape != (self.address_dim,) or mask.shape != (self.address_dim,):
            raise ValueError("partial_key and mask must match address_dim")
        padded = np.where(mask, partial_key, 0)
        return self.read(padded)

    def clear(self) -> None:
        self.counters[:] = 0


# ----------------------------------------------------------------------------
# Higher-level: bind context to content (hippocampal-style)
# ----------------------------------------------------------------------------

@dataclass
class HippocampalMemory:
    """A hippocampal-style memory: stores (context, content) pairs and
    retrieves content from context (or partial context).

    Internally uses an SDM with address_dim = context_dim and
    content_dim = content_dim.
    """
    context_dim: int = 32
    content_dim: int = 32
    n_addresses: int = 100
    k_nearest:   int = 8
    seed:        int = 0

    sdm: SparseDistributedMemory = field(default=None)

    def __post_init__(self) -> None:
        if self.sdm is None:
            self.sdm = SparseDistributedMemory(
                address_dim=self.context_dim,
                content_dim=self.content_dim,
                n_addresses=self.n_addresses,
                k_nearest=self.k_nearest,
                seed=self.seed,
            )

    def store(self, context: np.ndarray, content: np.ndarray) -> None:
        self.sdm.write(context, content)

    def recall(self, context: np.ndarray) -> np.ndarray:
        return self.sdm.read(context)

    def recall_partial(self, partial_context: np.ndarray,
                         known_mask: np.ndarray) -> np.ndarray:
        return self.sdm.recall_with_partial_cue(partial_context, known_mask)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def random_bipolar(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a uniformly-random ±1 vector."""
    return rng.choice([-1, 1], size=n).astype(np.float64)


def hamming_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of agreeing entries for two ±1 vectors."""
    if a.shape != b.shape:
        raise ValueError("shapes don't match")
    return float((a == b).mean())


def corrupt_bipolar(v: np.ndarray, p_flip: float,
                      rng: np.random.Generator) -> np.ndarray:
    """Flip each entry of a ±1 vector with probability p_flip."""
    flip = rng.uniform(size=v.shape) < p_flip
    return np.where(flip, -v, v)
